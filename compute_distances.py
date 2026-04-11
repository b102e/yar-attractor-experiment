"""
compute_distances.py
Computes pairwise cosine distances and runs statistical tests.
"""

import logging
import numpy as np
from itertools import combinations
from typing import Dict, List, Tuple

from scipy import stats
from sklearn.metrics.pairwise import cosine_distances

logger = logging.getLogger(__name__)


# ── Distance helpers ──────────────────────────────────────────────────────────

def pairwise_cosine(vecs: List[np.ndarray]) -> np.ndarray:
    """Returns full pairwise cosine distance matrix."""
    M = np.stack(vecs)
    return cosine_distances(M)


def within_group_distances(vecs: List[np.ndarray]) -> np.ndarray:
    """All unique pairwise cosine distances within a group."""
    D = pairwise_cosine(vecs)
    n = len(vecs)
    idx = np.triu_indices(n, k=1)
    return D[idx]


def between_group_distances(
    vecs_a: List[np.ndarray],
    vecs_b: List[np.ndarray],
) -> np.ndarray:
    """All pairwise cosine distances between two groups."""
    A = np.stack(vecs_a)
    B = np.stack(vecs_b)
    D = cosine_distances(A, B)
    return D.flatten()


def centroid_distance(
    vecs_a: List[np.ndarray],
    vec_b: np.ndarray,
) -> float:
    """Cosine distance from centroid of group A to a single vector B."""
    centroid = np.mean(np.stack(vecs_a), axis=0, keepdims=True)
    return float(cosine_distances(centroid, vec_b.reshape(1, -1))[0, 0])


# ── Bootstrap CI ──────────────────────────────────────────────────────────────

def bootstrap_ci(
    data: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
) -> Tuple[float, float]:
    rng = np.random.default_rng(42)
    means = [rng.choice(data, size=len(data), replace=True).mean()
             for _ in range(n_bootstrap)]
    alpha = (1 - ci) / 2
    lo, hi = np.percentile(means, [alpha * 100, (1 - alpha) * 100])
    return float(lo), float(hi)


# ── Statistical test ──────────────────────────────────────────────────────────

def compare_conditions(
    within_AB: np.ndarray,
    between_AB_C: np.ndarray,
    alpha_bonferroni: float = 0.0167,
    n_bootstrap: int = 1000,
) -> Dict:
    """
    One-sided Welch t-test: H1: within_AB < between_AB_C
    Bonferroni-corrected threshold: alpha_bonferroni = 0.05 / 3 layers
    """
    t_stat, p_value = stats.ttest_ind(
        within_AB, between_AB_C, alternative="less", equal_var=False
    )

    # Cohen's d (pooled SD)
    pooled = np.concatenate([within_AB, between_AB_C])
    d = (between_AB_C.mean() - within_AB.mean()) / pooled.std()

    ci_within = bootstrap_ci(within_AB, n_bootstrap)
    ci_between = bootstrap_ci(between_AB_C, n_bootstrap)

    return {
        "t_statistic":    float(t_stat),
        "p_value":        float(p_value),
        "cohens_d":       float(d),
        "significant":    bool(p_value < alpha_bonferroni),
        "alpha_used":     alpha_bonferroni,
        "mean_within_AB": float(within_AB.mean()),
        "std_within_AB":  float(within_AB.std()),
        "mean_between":   float(between_AB_C.mean()),
        "std_between":    float(between_AB_C.std()),
        "ci_within_AB":   list(ci_within),
        "ci_between":     list(ci_between),
        "n_within":       len(within_AB),
        "n_between":      len(between_AB_C),
    }


# ── Main computation ──────────────────────────────────────────────────────────

def compute_all_distances(
    activations: Dict[str, Dict[int, List[np.ndarray]]],
    layers: List[int],
    n_bootstrap: int = 1000,
    alpha_bonferroni: float = 0.0167,
) -> Dict:
    """
    For each layer:
      - within A+B distances
      - between (A+B) vs C distances
      - distance from D to centroid of A+B
      - statistical test
    """
    results = {}

    for layer in layers:
        logger.info(f"Computing distances at layer {layer}")

        vecs_A = activations["A"][layer]
        vecs_B = activations["B"][layer]
        vecs_C = activations["C"][layer]
        vecs_D = activations["D"][layer]

        vecs_AB = vecs_A + vecs_B

        # Core distances
        within_AB = within_group_distances(vecs_AB)
        between   = between_group_distances(vecs_AB, vecs_C)

        # D vs centroid of AB (exploratory)
        d_dist = centroid_distance(vecs_AB, vecs_D[0])

        # Stats
        test = compare_conditions(
            within_AB, between,
            alpha_bonferroni=alpha_bonferroni,
            n_bootstrap=n_bootstrap,
        )

        results[layer] = {
            "within_AB_raw":     within_AB.tolist(),
            "between_raw":       between.tolist(),
            "d_to_centroid_AB":  d_dist,
            "stats":             test,
        }

        sig = "✓ SIGNIFICANT" if test["significant"] else "✗ not significant"
        logger.info(
            f"  Layer {layer}: "
            f"within={test['mean_within_AB']:.4f} "
            f"between={test['mean_between']:.4f} "
            f"p={test['p_value']:.4f} "
            f"d={test['cohens_d']:.3f} "
            f"{sig}"
        )

    return results
