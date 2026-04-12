#!/usr/bin/env python3
"""
run.py — Condition C' ablation runner
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


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def model_tag(model_name: str) -> str:
    if "llama" in model_name.lower():
        return "llama"
    if "gemma" in model_name.lower():
        return "gemma"
    return "model"


def default_baseline_json(model_name: str) -> str:
    if "gemma" in model_name.lower():
        return CONFIG.baseline_results_gemma
    return CONFIG.baseline_results_llama


def load_baseline_between_c(path: str, layers: list[int]) -> dict[int, np.ndarray]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Baseline JSON not found: {p}")

    with p.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    out: dict[int, np.ndarray] = {}
    for layer in layers:
        layer_key = str(layer)
        if layer_key not in payload["results"]:
            raise KeyError(f"Layer {layer} missing in baseline JSON")

        layer_obj = payload["results"][layer_key]
        # original key in main experiment
        if "between_raw" in layer_obj:
            arr = layer_obj["between_raw"]
        elif "between_C_raw" in layer_obj:
            arr = layer_obj["between_C_raw"]
        else:
            raise KeyError(f"No baseline C distances in layer {layer}")

        out[layer] = np.array(arr, dtype=float)
    return out


def save_results(distance_results: dict, data: dict, out_dir: Path, model_name: str, baseline_json: str):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = out_dir / f"yar_attractor_cprime_{timestamp}.json"

    output = {
        "metadata": {
            "experiment": CONFIG.experiment_name,
            "timestamp": datetime.now().isoformat(),
            "model": model_name,
            "pooling": CONFIG.pooling,
            "layers": CONFIG.layers,
            "conditions": CONFIG.conditions,
            "seed": CONFIG.seed,
            "n_A": len(data["A"]),
            "n_B": len(data["B"]),
            "n_C_prime": len(data["C_prime"]),
            "n_within_expected": 28,
            "n_between_cprime_expected": 24,
            "alpha_bonferroni": CONFIG.alpha_bonferroni,
            "n_permutations": CONFIG.n_bootstrap,
            "baseline_between_c_json": baseline_json,
        },
        "results": distance_results,
    }

    with filename.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    return filename


def main():
    parser = argparse.ArgumentParser(description="YAR C' ablation")
    parser.add_argument("--model", default=CONFIG.model_name)
    parser.add_argument("--baseline-json", default=None)
    parser.add_argument("--skip-extraction", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    model_name = args.model
    baseline_json = args.baseline_json or default_baseline_json(model_name)
    tag = model_tag(model_name)

    out_dir = Path(CONFIG.results_dir) / "c_prime" / tag
    act_dir = out_dir / "activations"
    out_dir.mkdir(parents=True, exist_ok=True)
    act_dir.mkdir(parents=True, exist_ok=True)

    # logging per model output folder
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(out_dir / "experiment.log"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    logger = logging.getLogger(__name__)

    set_seed(CONFIG.seed)
    logger.info("Experiment: %s", CONFIG.experiment_name)
    logger.info("Model: %s", model_name)
    logger.info("Layers: %s", CONFIG.layers)
    logger.info("Pooling: %s", CONFIG.pooling)
    logger.info("Baseline JSON: %s", baseline_json)

    data = load_all(CONFIG.data_dir)
    if not verify_data(data):
        logger.error("Data verification failed")
        sys.exit(1)

    baseline_between_c = load_baseline_between_c(baseline_json, CONFIG.layers)

    if args.dry_run:
        logger.info("Dry run complete. Data and baseline JSON are valid.")
        return

    if args.skip_extraction:
        logger.info("Reloading activations from disk...")
        n_per = {k: len(v) for k, v in data.items()}
        activations = load_activations_from_disk(
            str(act_dir), list(data.keys()), CONFIG.layers, n_per
        )
    else:
        logger.info("Loading model...")
        model, tokenizer = load_model(model_name)
        activations = extract_all(
            model,
            tokenizer,
            data,
            CONFIG.layers,
            save_dir=str(act_dir),
            pooling=CONFIG.pooling,
        )
        del model
        torch.cuda.empty_cache()

    logger.info("Computing distances and statistics...")
    distance_results = compute_all_distances(
        activations=activations,
        baseline_between_c=baseline_between_c,
        layers=CONFIG.layers,
        n_bootstrap=CONFIG.n_bootstrap,
        alpha_bonferroni=CONFIG.alpha_bonferroni,
    )

    result_file = save_results(
        distance_results=distance_results,
        data=data,
        out_dir=out_dir,
        model_name=model_name,
        baseline_json=baseline_json,
    )
    logger.info("Results saved: %s", result_file)
    logger.info("Experiment complete.")


if __name__ == "__main__":
    main()
