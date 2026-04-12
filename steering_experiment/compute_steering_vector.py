"""
compute_steering_vector.py
Вычисляет steering vector из существующих .npy активаций.
Не требует GPU — работает с уже сохранёнными файлами.

Использование:
    python compute_steering_vector.py \
        --activations_dir results/llama/activations \
        --layer 24 \
        --output steering_vectors/llama_layer24.npy

Выводит:
    - delta вектор (centroid_AB - centroid_C)
    - нормализованный delta_normalized
    - косинусное расстояние между центроидами
    - статистику для проверки
"""

import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_distances
import argparse
import json


def load_condition_activations(activations_dir: Path, layer: int, condition: str):
    """Загружает все активации для условия на заданном слое."""
    vecs = []
    for f in sorted(activations_dir.glob(f"*_layer{layer}.npy")):
        name = f.stem.replace(f"_layer{layer}", "")
        if condition == "AB":
            if name in ("condition_A", "A_00") or name.startswith("B_"):
                vecs.append(np.load(f).flatten())
        elif condition == "C":
            if name.startswith("C_") and not name.startswith("C_prime") and not name.startswith("C_hybrid"):
                vecs.append(np.load(f).flatten())
    return np.stack(vecs) if vecs else None


def main(activations_dir: str, layer: int, output_path: str):
    acts_dir = Path(activations_dir)
    output = Path(output_path)
    output.parent.mkdir(exist_ok=True, parents=True)

    print(f"Loading activations from: {acts_dir}")
    print(f"Layer: {layer}")

    # Загрузить A+B активации
    ab_vecs = load_condition_activations(acts_dir, layer, "AB")
    c_vecs = load_condition_activations(acts_dir, layer, "C")

    if ab_vecs is None or c_vecs is None:
        print("ERROR: Could not load activations. Check directory and file naming.")
        return

    print(f"  A+B documents: {len(ab_vecs)}")
    print(f"  C documents:   {len(c_vecs)}")

    # Вычислить центроиды
    centroid_AB = ab_vecs.mean(axis=0)   # (4096,)
    centroid_C  = c_vecs.mean(axis=0)    # (4096,)

    # Steering vector
    delta = centroid_AB - centroid_C
    delta_norm = np.linalg.norm(delta)
    delta_normalized = delta / delta_norm

    # Проверка
    cos_dist = cosine_distances(
        centroid_AB.reshape(1, -1),
        centroid_C.reshape(1, -1)
    )[0][0]

    print(f"\nSteering vector stats:")
    print(f"  ||centroid_AB||   = {np.linalg.norm(centroid_AB):.4f}")
    print(f"  ||centroid_C||    = {np.linalg.norm(centroid_C):.4f}")
    print(f"  ||delta||         = {delta_norm:.4f}")
    print(f"  Cosine dist(AB,C) = {cos_dist:.4f}")
    print(f"  Vector shape      = {delta_normalized.shape}")

    # Сохранить
    np.save(output, delta_normalized)
    print(f"\nSaved: {output}")

    # Сохранить метаданные
    meta = {
        "layer": layer,
        "activations_dir": str(activations_dir),
        "n_AB": len(ab_vecs),
        "n_C": len(c_vecs),
        "delta_norm": float(delta_norm),
        "cosine_dist_AB_C": float(cos_dist),
        "vector_shape": list(delta_normalized.shape),
        "note": "delta_normalized = (centroid_AB - centroid_C) / ||centroid_AB - centroid_C||"
    }
    meta_path = output.with_suffix(".json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata: {meta_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--activations_dir", required=True)
    parser.add_argument("--layer", type=int, default=24)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    main(args.activations_dir, args.layer, args.output)
