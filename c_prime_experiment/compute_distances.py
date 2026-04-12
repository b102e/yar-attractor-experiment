"""
compute_distances.py
C' ablation: compares A+B within distances to A+B x C' distances,
and compares C' distances to baseline A+B x C distances from main results JSON.
"""

import logging
import numpy as np
from typing import Dict, List

from scipy import stats
from sklearn.metrics.pairwise import cosine_distances

logger = logging.getLogger(__name__)


def pairwise_cosine(vecs: List[np.ndarray]) -> np.ndarray:
    m = np.stack(vecs)
    return cosine_distances(m)


def within_group_distances(vecs: List[np.ndarray]) -> np.ndarray:
    d = pairwise_cosine(vecs)
    idx = np.triu_indices(len(vecs), k=1)
    return d[idx]


def between_group_distances(vecs_a: List[np.ndarray], vecs_b: List[np.ndarray]) -> np.ndarray:
    a = np.stack(vecs_a)
    b = np.stack(vecs_b)
    return cosine_distances(a, b).flatten()


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    pooled = np.concatenate([a, b])
    std = pooled.std()
    if std == 0:
        return 0.0
    return float((b.mean() - a.mean()) / std)


def permutation_test(a: np.ndarray, b: np.ndarray, n_permutations: int = 10000, seed: int = 42) -> float:
    """One-sided: H1 mean(a) < mean(b)."""
    rng = np.random.default_rng(seed)
    observed = b.mean() - a.mean()
    combined = np.concatenate([a, b]).copy()
    n_a = len(a)

    extreme = 0
    for _ in range(n_permutations):
        rng.shuffle(combined)
        pa = combined[:n_a]
        pb = combined[n_a:]
        if (pb.mean() - pa.mean()) >= observed:
            extreme += 1
    return float(extreme / n_permutations)


def compare_within_vs_between(within: np.ndarray, between: np.ndarray, alpha: float, n_perm: int) -> Dict:
    t_stat, p_welch = stats.ttest_ind(within, between, alternative="less", equal_var=False)
    u_stat, p_mw = stats.mannwhitneyu(within, between, alternative="less")
    p_perm = permutation_test(within, between, n_permutations=n_perm, seed=42)

    return {
        "mean_within_AB": float(within.mean()),
        "std_within_AB": float(within.std()),
        "mean_between_C_prime": float(between.mean()),
        "std_between_C_prime": float(between.std()),
        "n_within": int(len(within)),
        "n_between_cprime": int(len(between)),
        "welch_t": float(t_stat),
        "welch_p": float(p_welch),
        "permutation_p": float(p_perm),
        "mann_whitney_u": float(u_stat),
        "mann_whitney_p": float(p_mw),
        "cohens_d": cohens_d(within, between),
        "alpha_bonferroni": float(alpha),
        "significant_all": bool((p_welch < alpha) and (p_perm < alpha) and (p_mw < alpha)),
    }


def compare_cprime_vs_baseline_c(cprime: np.ndarray, baseline_c: np.ndarray) -> Dict:
    """Two-sided Welch t-test: expected non-significant if C' ~= C."""
    t_stat, p_val = stats.ttest_ind(cprime, baseline_c, alternative="two-sided", equal_var=False)
    return {
        "mean_cprime": float(cprime.mean()),
        "std_cprime": float(cprime.std()),
        "mean_baseline_C": float(baseline_c.mean()),
        "std_baseline_C": float(baseline_c.std()),
        "n_cprime": int(len(cprime)),
        "n_baseline_C": int(len(baseline_c)),
        "welch_t_two_sided": float(t_stat),
        "welch_p_two_sided": float(p_val),
        "delta_mean_cprime_minus_baseline_C": float(cprime.mean() - baseline_c.mean()),
    }


def compute_all_distances(
    activations: Dict[str, Dict[int, List[np.ndarray]]],
    baseline_between_c: Dict[int, np.ndarray],
    layers: List[int],
    n_bootstrap: int = 10000,
    alpha_bonferroni: float = 0.0167,
) -> Dict:
    results: Dict[int, Dict] = {}

    for layer in layers:
        vecs_ab = activations["A"][layer] + activations["B"][layer]
        vecs_cprime = activations["C_prime"][layer]

        within_ab = within_group_distances(vecs_ab)               # 28
        between_cprime = between_group_distances(vecs_ab, vecs_cprime)  # 24

        if layer not in baseline_between_c:
            raise KeyError(f"Missing baseline between_C distances for layer {layer}")

        between_c = baseline_between_c[layer]

        stats_main = compare_within_vs_between(
            within_ab,
            between_cprime,
            alpha=alpha_bonferroni,
            n_perm=n_bootstrap,
        )
        stats_cprime_vs_c = compare_cprime_vs_baseline_c(between_cprime, between_c)

        results[layer] = {
            "within_AB_raw": within_ab.tolist(),
            "between_C_prime_raw": between_cprime.tolist(),
            "between_C_baseline_raw": between_c.tolist(),
            "stats_within_vs_cprime": stats_main,
            "stats_cprime_vs_c": stats_cprime_vs_c,
        }

        logger.info(
            "Layer %s | within=%.4f cprime=%.4f baselineC=%.4f | p_welch=%.3e p_perm=%.4f p_mw=%.3e | p(cprime_vs_C)=%.3e",
            layer,
            stats_main["mean_within_AB"],
            stats_main["mean_between_C_prime"],
            stats_cprime_vs_c["mean_baseline_C"],
            stats_main["welch_p"],
            stats_main["permutation_p"],
            stats_main["mann_whitney_p"],
            stats_cprime_vs_c["welch_p_two_sided"],
        )

    return results
