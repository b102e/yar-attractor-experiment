"""
bootstrap_analysis.py
Загружает активации A+B и 30 D_random сэмплов,
вычисляет распределение расстояний до центроида A+B,
сравнивает с D_distilled.
"""

import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_distances
import json


def analyze(activations_dir: str, model_label: str, layers=[8, 16, 24]):
    acts_dir = Path(activations_dir)

    results = {}
    for layer in layers:
        ab_vecs = []
        for f in sorted(acts_dir.glob(f"*_layer{layer}.npy")):
            name = f.stem.replace(f"_layer{layer}", "")
            if name in {"A_00", "condition_A"} or name.startswith("B_"):
                ab_vecs.append(np.load(f).flatten())

        if not ab_vecs:
            raise RuntimeError(f"No A/B vectors found in {acts_dir} for layer {layer}")

        centroid_ab = np.mean(ab_vecs, axis=0, keepdims=True)

        d_candidates = [
            acts_dir / f"D_00_layer{layer}.npy",
            acts_dir / f"condition_D_layer{layer}.npy",
        ]
        d_dist_file = next((p for p in d_candidates if p.exists()), None)
        if d_dist_file is None:
            raise RuntimeError(f"No distilled D activation file for layer {layer}")

        d_dist = np.load(d_dist_file).flatten().reshape(1, -1)
        dist_distilled = cosine_distances(d_dist, centroid_ab)[0][0]

        random_dists = []
        for f in sorted(acts_dir.glob(f"D_random_*_layer{layer}.npy")):
            vec = np.load(f).flatten().reshape(1, -1)
            d = cosine_distances(vec, centroid_ab)[0][0]
            random_dists.append(d)

        if not random_dists:
            raise RuntimeError(f"No D_random_* files for layer {layer} in {acts_dir}")

        results[layer] = {
            "D_distilled": float(dist_distilled),
            "D_random_mean": float(np.mean(random_dists)),
            "D_random_median": float(np.median(random_dists)),
            "D_random_std": float(np.std(random_dists)),
            "D_random_min": float(np.min(random_dists)),
            "D_random_max": float(np.max(random_dists)),
            "n_samples": int(len(random_dists)),
            "distilled_beats_random_pct": float(
                sum(1 for d in random_dists if d > dist_distilled) / len(random_dists) * 100
            ),
        }

        print(f"\n{model_label} | Layer {layer}:")
        print(f"  D_distilled:          {dist_distilled:.4f}")
        print(f"  D_random mean±std:    {np.mean(random_dists):.4f} ± {np.std(random_dists):.4f}")
        print(f"  D_random range:       [{np.min(random_dists):.4f}, {np.max(random_dists):.4f}]")
        print(
            f"  D_distilled < D_random in {results[layer]['distilled_beats_random_pct']:.0f}% of samples"
        )

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--activations_dir", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    results = analyze(args.activations_dir, args.model)

    out = args.activations_dir.replace("activations", f"bootstrap_{args.model.replace('/', '_')}.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out}")
