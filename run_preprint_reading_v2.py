import torch
import numpy as np
import json
import argparse
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics.pairwise import cosine_distances
from datetime import datetime

NEUTRAL_PROMPT = "Привет. Как дела?"


def load_centroid(activations_dir, layer):
    acts_dir = Path(activations_dir)
    vecs = []
    for f in sorted(acts_dir.glob(f"*_layer{layer}.npy")):
        name = f.stem.replace(f"_layer{layer}", "")
        if name in ("condition_A", "A_00") or name.startswith("B_"):
            vecs.append(np.load(f).flatten())
    if not vecs:
        raise FileNotFoundError(f"No A+B activations in {acts_dir} for layer {layer}")
    centroid = np.mean(vecs, axis=0, keepdims=True)
    print(f"  Layer {layer}: {len(vecs)} A+B vectors loaded")
    return centroid


def extract_activation(model, tokenizer, text, layers, device, max_length=4096):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_length).to(device)
    n_tokens = inputs['input_ids'].shape[1]
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    result = {}
    for layer in layers:
        hidden = outputs.hidden_states[layer][0]
        result[layer] = hidden.mean(dim=0).cpu().float().numpy()
    return result, n_tokens


def main(model_name, cognitive_core_path, preprint_path, sham_preprint_path,
         activations_dir, output_dir, layers, max_length):

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    cognitive_core = Path(cognitive_core_path).read_text(encoding='utf-8')
    preprint = Path(preprint_path).read_text(encoding='utf-8')
    sham = Path(sham_preprint_path).read_text(encoding='utf-8')

    print(f"Cognitive core:  {len(cognitive_core.split())} words")
    print(f"YAR preprint:    {len(preprint.split())} words")
    print(f"Sham preprint:   {len(sham.split())} words")

    print("\nLoading A+B centroids...")
    centroids = {layer: load_centroid(activations_dir, layer) for layer in layers}

    print(f"\nLoading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map='auto')
    model.eval()
    print(f"Loaded. {sum(p.numel() for p in model.parameters())/1e9:.1f}B params")

    conditions = {
        "baseline_empty": NEUTRAL_PROMPT,
        "baseline_core": cognitive_core,
        "preprint_only": preprint,
        "core_plus_preprint": cognitive_core + "\n\n" + "=" * 60 + "\n\n" + preprint,
        "sham_preprint_only": sham,
    }

    results = {}
    for cond_name, text in conditions.items():
        print(f"\n--- {cond_name} ---")
        acts, n_tokens = extract_activation(model, tokenizer, text, layers, device, max_length)
        print(f"  Tokens: {n_tokens}")
        layer_results = {}
        for layer in layers:
            dist = float(cosine_distances(acts[layer].reshape(1,-1), centroids[layer])[0][0])
            print(f"  Layer {layer}: dist_to_attractor = {dist:.4f}")
            layer_results[str(layer)] = {"dist_to_centroid_AB": dist, "n_tokens": n_tokens}
        results[cond_name] = layer_results

    print("\n" + "=" * 65)
    print(f"{'Condition':<30} {'L8':>8} {'L16':>8} {'L24':>8}")
    print("-" * 55)
    for cond in conditions:
        d = [results[cond][str(l)]["dist_to_centroid_AB"] for l in layers]
        print(f"{cond:<30} {d[0]:>8.4f} {d[1]:>8.4f} {d[2]:>8.4f}")

    print("\nHYPOTHESIS CHECKS:")
    for layer in layers:
        r = {c: results[c][str(layer)]["dist_to_centroid_AB"] for c in conditions}
        print(f"\nLayer {layer}:")
        print(f"  H_A/B: core+preprint {r['core_plus_preprint']:.4f} vs core {r['baseline_core']:.4f} => {'H_A (closer)' if r['core_plus_preprint'] < r['baseline_core'] else 'H_B (farther)'}")
        print(f"  H_C:   preprint_only {r['preprint_only']:.4f} vs empty {r['baseline_empty']:.4f} => {'SUPPORTED' if r['preprint_only'] < r['baseline_empty'] else 'not supported'}")
        print(f"  H_C_s: yar {r['preprint_only']:.4f} vs sham {r['sham_preprint_only']:.4f} => {'YAR-SPECIFIC' if r['preprint_only'] < r['sham_preprint_only'] else 'not specific (sham equally close)'}")

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_short = model_name.split('/')[-1].replace('-','_').lower()
    output = {
        "metadata": {
            "experiment": "preprint_reading_v2",
            "timestamp": datetime.now().isoformat(),
            "model": model_name,
            "conditions": list(conditions.keys()),
            "layers": layers,
            "max_length": max_length,
            "activations_dir": str(activations_dir),
            "sham_source": "arxiv:2505.17237 (protein folding dynamics)"
        },
        "results": results
    }
    out_path = output_dir / f"preprint_reading_{model_short}_{ts}.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--cognitive_core', required=True)
    parser.add_argument('--preprint', required=True)
    parser.add_argument('--sham_preprint', required=True)
    parser.add_argument('--activations_dir', required=True)
    parser.add_argument('--output_dir', default='results/preprint_reading')
    parser.add_argument('--layers', nargs='+', type=int, default=[8, 16, 24])
    parser.add_argument('--max_length', type=int, default=4096)
    args = parser.parse_args()
    torch.manual_seed(42)
    main(args.model, args.cognitive_core, args.preprint, args.sham_preprint,
         args.activations_dir, args.output_dir, args.layers, args.max_length)
