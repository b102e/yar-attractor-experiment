"""
permutation_test.py
Permutation test (non-parametric) для проверки H1 из существующих JSON результатов.
Не требует GPU — работает с уже сохранёнными расстояниями.

Использование:
    python permutation_test.py --json results/yar_attractor_v1_20260411_152017.json
    python permutation_test.py --json results/yar_attractor_v1_20260411_160259.json

Выводит:
    - Permutation p-value для каждого слоя
    - Mann-Whitney U p-value для каждого слоя
    - Сравнение с Welch t-test из JSON
"""

import json
import argparse
import numpy as np
from scipy import stats


def permutation_test(within, between, n_permutations=10000, seed=42):
    """
    One-sided permutation test: H1: mean(within) < mean(between)
    Объединяем все расстояния, случайно разбиваем на две группы
    того же размера, считаем долю случаев где разница >= наблюдаемой.
    """
    rng = np.random.default_rng(seed)

    within = np.array(within)
    between = np.array(between)

    observed_diff = np.mean(between) - np.mean(within)

    combined = np.concatenate([within, between])
    n_within = len(within)

    count_extreme = 0
    for _ in range(n_permutations):
        rng.shuffle(combined)
        perm_within = combined[:n_within]
        perm_between = combined[n_within:]
        perm_diff = np.mean(perm_between) - np.mean(perm_within)
        if perm_diff >= observed_diff:
            count_extreme += 1

    p_value = count_extreme / n_permutations
    return p_value, observed_diff


def mann_whitney_test(within, between):
    """
    Mann-Whitney U test (one-sided): H1: within < between
    """
    stat, p_two_sided = stats.mannwhitneyu(within, between, alternative='less')
    return stat, p_two_sided


def main(json_path, n_permutations=10000):
    with open(json_path) as f:
        data = json.load(f)

    model = data['metadata']['model']
    layers = data['metadata']['layers']
    alpha = data['metadata']['alpha_bonferroni']

    print(f"\nModel: {model}")
    print(f"Layers: {layers}")
    print(f"Bonferroni alpha: {alpha}")
    print(f"Permutations: {n_permutations}")
    print("=" * 70)

    results_summary = []

    for layer in layers:
        layer_data = data['results'][str(layer)]
        within = layer_data['within_AB_raw']
        between = layer_data['between_raw']

        # Welch t-test (from JSON)
        welch_p = layer_data['stats']['p_value']
        cohens_d = layer_data['stats']['cohens_d']
        mean_within = layer_data['stats']['mean_within_AB']
        mean_between = layer_data['stats']['mean_between']

        # Permutation test
        perm_p, obs_diff = permutation_test(within, between, n_permutations)

        # Mann-Whitney U
        mw_stat, mw_p = mann_whitney_test(within, between)

        sig_perm = "✓" if perm_p < alpha else "✗"
        sig_mw = "✓" if mw_p < alpha else "✗"

        print(f"\nLayer {layer}:")
        print(f"  Mean within:   {mean_within:.5f}")
        print(f"  Mean between:  {mean_between:.5f}")
        print(f"  Cohen's d:     {cohens_d:.3f}")
        print(f"  Observed diff: {obs_diff:.5f}")
        print(f"  Welch t-test:  p = {welch_p:.2e}  {'✓' if welch_p < alpha else '✗'}")
        print(f"  Permutation:   p = {perm_p:.4f}  {sig_perm}  (n={n_permutations})")
        print(f"  Mann-Whitney:  p = {mw_p:.2e}  {sig_mw}  (U={mw_stat:.0f})")

        results_summary.append({
            "layer": layer,
            "mean_within": mean_within,
            "mean_between": mean_between,
            "cohens_d": cohens_d,
            "welch_p": welch_p,
            "permutation_p": perm_p,
            "mann_whitney_p": float(mw_p),
            "mann_whitney_U": float(mw_stat),
            "significant_permutation": perm_p < alpha,
            "significant_mann_whitney": mw_p < alpha
        })

    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print(f"{'Layer':<8} {'Cohen d':<10} {'Welch p':<14} {'Permut p':<12} {'MW p':<14} {'Sig?'}")
    print("-" * 70)
    for r in results_summary:
        all_sig = all([r['welch_p'] < alpha, r['permutation_p'] < alpha, r['mann_whitney_p'] < alpha])
        sig_str = "ALL ✓" if all_sig else "MIXED"
        print(f"  {r['layer']:<6} {r['cohens_d']:<10.3f} {r['welch_p']:<14.2e} {r['permutation_p']:<12.4f} {r['mann_whitney_p']:<14.2e} {sig_str}")

    # Save results
    out_path = json_path.replace('.json', '_permutation_results.json')
    output = {
        "model": model,
        "n_permutations": n_permutations,
        "alpha_bonferroni": alpha,
        "layers": results_summary
    }
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True, help="Path to experiment JSON file")
    parser.add_argument("--n_permutations", type=int, default=10000)
    args = parser.parse_args()
    main(args.json, args.n_permutations)
