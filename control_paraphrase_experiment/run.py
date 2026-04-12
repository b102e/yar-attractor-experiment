#!/usr/bin/env python3
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


def model_tag(model_name: str) -> str:
    m = model_name.lower()
    if "llama" in m:
        return "llama"
    if "gemma" in m:
        return "gemma"
    return "model"


def default_baseline_json(model_name: str) -> str:
    return CONFIG.baseline_results_gemma if "gemma" in model_name.lower() else CONFIG.baseline_results_llama


def load_baseline_within_yar(path: str, layers: list[int]) -> dict[int, float]:
    p = Path(path)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    out = {}
    for layer in layers:
        lo = payload["results"].get(str(layer), {})
        stats = lo.get("stats") or lo.get("stats_C")
        if stats and "mean_within_AB" in stats:
            out[layer] = float(stats["mean_within_AB"])
    return out


def save_results(results: dict, data: dict, out_dir: Path, model_name: str, baseline_json: str):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"control_paraphrase_{ts}.json"
    payload = {
        "metadata": {
            "experiment": CONFIG.experiment_name,
            "timestamp": datetime.now().isoformat(),
            "model": model_name,
            "pooling": CONFIG.pooling,
            "layers": CONFIG.layers,
            "seed": CONFIG.seed,
            "n_yar": len(data["YAR_A"]) + len(data["YAR_B"]),
            "n_sigma": len(data["SIGMA_ORIGINAL"]) + len(data["SIGMA_B"]),
            "pairs_within": 28,
            "pairs_between_yar_sigma": 64,
            "baseline_main_json": baseline_json,
        },
        "results": results,
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=CONFIG.model_name)
    parser.add_argument("--baseline-json", default=None)
    parser.add_argument("--skip-extraction", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    model_name = args.model
    baseline_json = args.baseline_json or default_baseline_json(model_name)
    tag = model_tag(model_name)

    out_dir = Path(CONFIG.results_dir) / tag
    act_dir = out_dir / "activations"
    out_dir.mkdir(parents=True, exist_ok=True)
    act_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(out_dir / "experiment.log"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    log = logging.getLogger(__name__)

    set_seed(CONFIG.seed)
    data = load_all(CONFIG.yar_data_dir, CONFIG.sigma_data_dir)
    if not verify_data(data):
        log.error("Data verification failed")
        sys.exit(1)

    if args.dry_run:
        log.info("Dry run complete")
        return

    if args.skip_extraction:
        n_per = {k: len(v) for k, v in data.items()}
        activations = load_activations_from_disk(str(act_dir), list(data.keys()), CONFIG.layers, n_per)
    else:
        model, tokenizer = load_model(model_name)
        activations = extract_all(model, tokenizer, data, CONFIG.layers, str(act_dir), pooling=CONFIG.pooling)
        del model
        torch.cuda.empty_cache()

    baseline_within = load_baseline_within_yar(baseline_json, CONFIG.layers)
    results = compute_all_distances(activations, CONFIG.layers, baseline_within_yar=baseline_within)
    path = save_results(results, data, out_dir, model_name, baseline_json)
    log.info("Saved results: %s", path)


if __name__ == "__main__":
    main()
