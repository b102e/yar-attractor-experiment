"""
run_truncation.py
Experiment: last-token pooling on TRUNCATED documents (first N tokens).
"""

import torch
import numpy as np
import json
import argparse
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics.pairwise import cosine_distances
from scipy import stats
from datetime import datetime


def load_condition(path: Path, tokenizer, max_tokens: int):
    text = path.read_text(encoding="utf-8")
    tokens = tokenizer.encode(text, add_special_tokens=True)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    return tokenizer.decode(tokens, skip_special_tokens=False), len(tokens)


def extract_activation(model, tokenizer, text: str, layers: list, pooling: str, device):
    inputs = tokenizer(text, return_tensors="pt", truncation=False).to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    result = {}
    for layer in layers:
        hidden = outputs.hidden_states[layer][0]
        if pooling == "mean":
            vec = hidden.mean(dim=0).cpu().float().numpy()
        elif pooling == "last":
            vec = hidden[-1].cpu().float().numpy()
        else:
            raise ValueError(f"Unknown pooling: {pooling}")
        result[layer] = vec

    return result


def compute_stats(within, between):
    t, p = stats.ttest_ind(within, between, equal_var=False, alternative="less")
    d = (np.mean(between) - np.mean(within)) / np.sqrt((np.std(within) ** 2 + np.std(between) ** 2) / 2)
    return {
        "mean_within": float(np.mean(within)),
        "std_within": float(np.std(within)),
        "mean_between": float(np.mean(between)),
        "std_between": float(np.std(between)),
        "welch_t": float(t),
        "welch_p": float(p),
        "cohens_d": float(d),
        "significant": bool(p < 0.0167),
        "n_within": len(within),
        "n_between": len(between),
    }


def main(model_name, data_dir, output_dir, layers, max_tokens, pooling):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print(f"Model: {model_name}")
    print(f"Pooling: {pooling} | Max tokens: {max_tokens}")
    print(f"Layers: {layers}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    model.eval()
    print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B")

    conditions = {}

    a_text, a_len = load_condition(data_dir / "condition_A.txt", tokenizer, max_tokens)
    conditions["A"] = [a_text]
    print(f"A: {a_len} tokens (truncated to {max_tokens})")

    b_texts = []
    for f in sorted((data_dir / "condition_B").glob("*.txt")):
        text, tlen = load_condition(f, tokenizer, max_tokens)
        b_texts.append(text)
        print(f"  {f.name}: {tlen} tokens")
    conditions["B"] = b_texts

    c_texts = []
    for f in sorted((data_dir / "condition_C").glob("*.txt")):
        text, _ = load_condition(f, tokenizer, max_tokens)
        c_texts.append(text)
    conditions["C"] = c_texts
    print(f"B: {len(b_texts)} files | C: {len(c_texts)} files")

    all_acts = {"A": [], "B": [], "C": []}
    for cond, texts in conditions.items():
        print(f"Extracting {cond}...")
        for text in texts:
            acts = extract_activation(model, tokenizer, text, layers, pooling, device)
            all_acts[cond].append(acts)

    results = {}
    for layer in layers:
        ab_vecs = [all_acts["A"][0][layer]] + [all_acts["B"][i][layer] for i in range(len(b_texts))]
        c_vecs = [all_acts["C"][i][layer] for i in range(len(c_texts))]

        ab_mat = np.stack(ab_vecs)
        c_mat = np.stack(c_vecs)

        within = []
        for i in range(len(ab_mat)):
            for j in range(i + 1, len(ab_mat)):
                d = cosine_distances(ab_mat[i : i + 1], ab_mat[j : j + 1])[0][0]
                within.append(float(d))

        between = []
        for ab in ab_mat:
            for c in c_mat:
                d = cosine_distances(ab.reshape(1, -1), c.reshape(1, -1))[0][0]
                between.append(float(d))

        stats_result = compute_stats(within, between)
        results[str(layer)] = {"within_raw": within, "between_raw": between, "stats": stats_result}

        sig = "✓ SIGNIFICANT" if stats_result["significant"] else "✗ not significant"
        print(
            f"  Layer {layer}: within={stats_result['mean_within']:.4f} "
            f"between={stats_result['mean_between']:.4f} "
            f"d={stats_result['cohens_d']:.3f} p={stats_result['welch_p']:.2e} {sig}"
        )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = model_name.split("/")[-1].replace("-", "_").lower()
    out = {
        "metadata": {
            "experiment": "truncation_pooling",
            "timestamp": datetime.now().isoformat(),
            "model": model_name,
            "pooling": pooling,
            "max_tokens": max_tokens,
            "layers": layers,
            "seed": 42,
            "n_A": 1,
            "n_B": len(b_texts),
            "n_C": len(c_texts),
            "alpha_bonferroni": 0.0167,
        },
        "results": results,
    }
    out_path = output_dir / f"truncation_{pooling}_{max_tokens}tok_{model_short}_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--output_dir", default="results/truncation")
    parser.add_argument("--layers", nargs="+", type=int, default=[8, 16, 24])
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--pooling", choices=["mean", "last"], default="last")
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)
    main(args.model, args.data_dir, args.output_dir, args.layers, args.max_tokens, args.pooling)
