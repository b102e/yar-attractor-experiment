import numpy as np
from scipy import stats
from sklearn.metrics.pairwise import cosine_distances


def within_group_distances(vecs):
    m = np.stack(vecs)
    d = cosine_distances(m)
    idx = np.triu_indices(len(vecs), k=1)
    return d[idx]


def between_group_distances(vecs_a, vecs_b):
    a = np.stack(vecs_a)
    b = np.stack(vecs_b)
    return cosine_distances(a, b).flatten()


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    pooled = np.concatenate([a, b])
    s = pooled.std()
    if s == 0:
        return 0.0
    return float((b.mean() - a.mean()) / s)


def layer_stats(within_yar: np.ndarray, within_sigma: np.ndarray):
    t, p = stats.ttest_ind(within_yar, within_sigma, equal_var=False)  # two-sided
    return {
        "mean_within_yar": float(within_yar.mean()),
        "std_within_yar": float(within_yar.std()),
        "mean_within_sigma": float(within_sigma.mean()),
        "std_within_sigma": float(within_sigma.std()),
        "n_within_yar": int(len(within_yar)),
        "n_within_sigma": int(len(within_sigma)),
        "welch_t": float(t),
        "welch_p_two_sided": float(p),
        "cohens_d_sigma_minus_yar": cohens_d(within_yar, within_sigma),
    }


def compute_all_distances(activations, layers, baseline_within_yar=None):
    results = {}
    for layer in layers:
        yar_cluster = activations["YAR_A"][layer] + activations["YAR_B"][layer]
        sigma_cluster = activations["SIGMA_ORIGINAL"][layer] + activations["SIGMA_B"][layer]

        within_yar = within_group_distances(yar_cluster)          # 28
        within_sigma = within_group_distances(sigma_cluster)      # 28
        between_yar_sigma = between_group_distances(yar_cluster, sigma_cluster)  # 64

        stats_layer = layer_stats(within_yar, within_sigma)
        if baseline_within_yar is not None and layer in baseline_within_yar:
            stats_layer["baseline_main_within_yar"] = float(baseline_within_yar[layer])
            stats_layer["delta_run_vs_main_within_yar"] = float(stats_layer["mean_within_yar"] - baseline_within_yar[layer])

        results[layer] = {
            "within_yar_raw": within_yar.tolist(),
            "within_sigma_raw": within_sigma.tolist(),
            "between_yar_sigma_raw": between_yar_sigma.tolist(),
            "stats": stats_layer,
            "between_summary": {
                "mean_between_yar_sigma": float(between_yar_sigma.mean()),
                "std_between_yar_sigma": float(between_yar_sigma.std()),
                "n_between_yar_sigma": int(len(between_yar_sigma)),
            },
        }
    return results
