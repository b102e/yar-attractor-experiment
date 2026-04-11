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
      - between (A+B) vs C_hybrid distances
      - distances from D and D_random to centroid of A+B
      - statistical test
    """
    results = {}

    for layer in layers:
        logger.info(f"Computing distances at layer {layer}")

        vecs_A = activations["A"][layer]
        vecs_B = activations["B"][layer]
        vecs_C = activations["C"][layer]
        vecs_C_hybrid = activations["C_hybrid"][layer]
        vecs_D = activations["D"][layer]
        vecs_D_random = activations["D_random"][layer]

        vecs_AB = vecs_A + vecs_B

        # Core distances
        within_AB = within_group_distances(vecs_AB)
        between_c = between_group_distances(vecs_AB, vecs_C)
        between_c_hybrid = between_group_distances(vecs_AB, vecs_C_hybrid)

        # D and D_random vs centroid of AB (exploratory)
        d_dist = centroid_distance(vecs_AB, vecs_D[0])
        d_random_dist = centroid_distance(vecs_AB, vecs_D_random[0])

        # Stats: AB vs C
        test_c = compare_conditions(
            within_AB, between_c,
            alpha_bonferroni=alpha_bonferroni,
            n_bootstrap=n_bootstrap,
        )
        # Stats: AB vs C_hybrid
        test_c_hybrid = compare_conditions(
            within_AB, between_c_hybrid,
            alpha_bonferroni=alpha_bonferroni,
            n_bootstrap=n_bootstrap,
        )

        results[layer] = {
            # Backward-compatible keys (original C and distilled D)
            "within_AB_raw":         within_AB.tolist(),
            "between_raw":           between_c.tolist(),
            "d_to_centroid_AB":      d_dist,
            "stats":                 test_c,
            # New ablation outputs
            "between_C_raw":         between_c.tolist(),
            "between_C_hybrid_raw":  between_c_hybrid.tolist(),
            "d_to_centroid_AB_D":    d_dist,
            "d_to_centroid_AB_D_random": d_random_dist,
            "stats_C":               test_c,
            "stats_C_hybrid":        test_c_hybrid,
            "ablation_summary": {
                "mean_between_C": float(between_c.mean()),
                "mean_between_C_hybrid": float(between_c_hybrid.mean()),
                "delta_C_hybrid_minus_C": float(between_c_hybrid.mean() - between_c.mean()),
                "mean_D_distilled": float(d_dist),
                "mean_D_random": float(d_random_dist),
                "delta_D_random_minus_D_distilled": float(d_random_dist - d_dist),
            },
        }

        sig_c = "✓ SIGNIFICANT" if test_c["significant"] else "✗ not significant"
        sig_c_h = "✓ SIGNIFICANT" if test_c_hybrid["significant"] else "✗ not significant"
        logger.info(
            f"  Layer {layer}: "
            f"within={test_c['mean_within_AB']:.4f} "
            f"between_C={test_c['mean_between']:.4f} "
            f"between_C_hybrid={test_c_hybrid['mean_between']:.4f} "
            f"p_C={test_c['p_value']:.4f} "
            f"p_C_hybrid={test_c_hybrid['p_value']:.4f} "
            f"d_C={test_c['cohens_d']:.3f} "
            f"d_C_hybrid={test_c_hybrid['cohens_d']:.3f} "
            f"D={d_dist:.4f} "
            f"D_random={d_random_dist:.4f} "
            f"[C:{sig_c}] [C_hybrid:{sig_c_h}]"
        )

    return results
