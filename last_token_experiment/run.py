#!/usr/bin/env python3
"""
run.py — Main experiment runner

Experiment: Identity as Attractor
Hypothesis: cognitive_core of a persistent agent induces a stable,
            specific region in LLM activation space analogous to
            concept attractors (Chytas & Singh, 2025).

Usage:
    python run.py                    # full run
    python run.py --skip-extraction  # reload saved activations
    python run.py --dry-run          # verify data only
"""

import argparse
import json
import logging
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from config import CONFIG
from data_loader import load_all, verify_data
from extract_activations import load_model, extract_all, load_activations_from_disk
from compute_distances import compute_all_distances
from visualize import generate_all


# ── Logging setup ─────────────────────────────────────────────────────────────

Path(CONFIG.results_dir).mkdir(parents=True, exist_ok=True)
Path(CONFIG.figures_dir).mkdir(parents=True, exist_ok=True)
Path(CONFIG.activations_dir).mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(f"{CONFIG.results_dir}/experiment.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


# ── Seed ──────────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Seed set: {seed}")


# ── Result saving ─────────────────────────────────────────────────────────────

def save_results(distance_results: dict, data: dict, model_name: str, pooling: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{CONFIG.results_dir}/{CONFIG.experiment_name}_{timestamp}.json"

    output = {
        "metadata": {
            "experiment":  CONFIG.experiment_name,
            "timestamp":   datetime.now().isoformat(),
            "model":       model_name,
            "pooling":     pooling,
            "layers":      CONFIG.layers,
            "conditions":  CONFIG.conditions,
            "seed":        CONFIG.seed,
            "n_A":         len(data["A"]),
            "n_B":         len(data["B"]),
            "n_C":         len(data["C"]),
            "n_C_hybrid":  len(data["C_hybrid"]),
            "n_D":         len(data["D"]),
            "n_D_random":  len(data["D_random"]),
            "alpha_bonferroni": CONFIG.alpha_bonferroni,
        },
        "results": distance_results,
    }

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info(f"Results saved: {filename}")
    return filename


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary(distance_results: dict):
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    print(
        f"{'Layer':<8} {'Within A+B':>12} {'Between C':>12} "
        f"{'Between C_h':>12} {'p(C)':>10} {'p(C_h)':>10}"
    )
    print("-" * 60)

    for layer in sorted(distance_results.keys()):
        s_c = distance_results[layer]["stats_C"]
        s_ch = distance_results[layer]["stats_C_hybrid"]
        print(
            f"{layer:<8} "
            f"{s_c['mean_within_AB']:>12.4f} "
            f"{s_c['mean_between']:>12.4f} "
            f"{s_ch['mean_between']:>12.4f} "
            f"{s_c['p_value']:>10.4f} "
            f"{s_ch['p_value']:>10.4f}"
        )

    print("=" * 60)
    print(f"Bonferroni-corrected α = {CONFIG.alpha_bonferroni}")

    # H2: convergence trend
    within_means = [distance_results[l]["stats_C"]["mean_within_AB"]
                    for l in sorted(distance_results.keys())]
    if within_means[-1] < within_means[0]:
        print("H2 (convergence): ✓ Within-group distance decreases with depth")
    else:
        print("H2 (convergence): ✗ No clear convergence trend")

    # Ablation: C vs C_hybrid
    print("\nAblation — A+B to C vs C_hybrid:")
    for layer in sorted(distance_results.keys()):
        ab = distance_results[layer]["ablation_summary"]
        print(
            f"  Layer {layer}: C={ab['mean_between_C']:.4f} | "
            f"C_hybrid={ab['mean_between_C_hybrid']:.4f} | "
            f"Δ(C_h-C)={ab['delta_C_hybrid_minus_C']:+.4f}"
        )

    # Ablation: D_distilled vs D_random
    print("\nAblation — D_distilled vs D_random to centroid(A+B):")
    for layer in sorted(distance_results.keys()):
        ab = distance_results[layer]["ablation_summary"]
        print(
            f"  Layer {layer}: D={ab['mean_D_distilled']:.4f} | "
            f"D_random={ab['mean_D_random']:.4f} | "
            f"Δ(D_rand-D)={ab['delta_D_random_minus_D_distilled']:+.4f}"
        )

    print("=" * 60 + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="YAR Attractor Experiment")
    parser.add_argument("--skip-extraction", action="store_true",
                        help="Reload activations from disk instead of re-running model")
    parser.add_argument("--dry-run", action="store_true",
                        help="Verify data only, do not run model")
    parser.add_argument("--model", default=None,
                        help="Model override (e.g. meta-llama/Llama-3.1-8B-Instruct)")
    parser.add_argument("--pooling", choices=["mean", "last"], default=None,
                        help="Pooling strategy for hidden states")
    args = parser.parse_args()

    model_name = args.model if args.model else CONFIG.model_name
    pooling = args.pooling if args.pooling else CONFIG.pooling

    set_seed(CONFIG.seed)

    logger.info(f"Experiment: {CONFIG.experiment_name}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Pooling: {pooling}")
    logger.info(f"Layers: {CONFIG.layers}")

    # 1. Load data
    logger.info("Loading data...")
    data = load_all(CONFIG.data_dir)
    ok = verify_data(data)
    if not ok:
        logger.error("Data verification failed. Aborting.")
        sys.exit(1)

    if args.dry_run:
        logger.info("Dry run complete. Data OK.")
        return

    # 2. Activations
    if args.skip_extraction:
        logger.info("Reloading activations from disk...")
        n_per = {c: len(texts) for c, texts in data.items()}
        activations = load_activations_from_disk(
            CONFIG.activations_dir, list(data.keys()), CONFIG.layers, n_per
        )
    else:
        logger.info("Loading model...")
        model, tokenizer = load_model(model_name)
        activations = extract_all(
            model, tokenizer, data, CONFIG.layers, CONFIG.activations_dir, pooling=pooling
        )
        del model  # free VRAM
        torch.cuda.empty_cache()

    # 3. Distances & statistics
    logger.info("Computing distances...")
    distance_results = compute_all_distances(
        activations,
        CONFIG.layers,
        n_bootstrap=CONFIG.n_bootstrap,
        alpha_bonferroni=CONFIG.alpha_bonferroni,
    )

    # 4. Save results
    save_results(distance_results, data, model_name=model_name, pooling=pooling)

    # 5. Figures
    logger.info("Generating figures...")
    generate_all(
        activations,
        distance_results,
        CONFIG.layers,
        CONFIG.figures_dir,
        CONFIG.tsne_perplexity,
    )

    # 6. Summary
    print_summary(distance_results)
    logger.info("Experiment complete.")


if __name__ == "__main__":
    main()
