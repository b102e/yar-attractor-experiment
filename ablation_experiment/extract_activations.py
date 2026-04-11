"""
extract_activations.py
Extracts hidden states from transformer layers for each condition text.
"""

import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

logger = logging.getLogger(__name__)


def load_model(
    model_name: str,
    device: str = "auto"
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    logger.info(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
    )
    model.eval()
    logger.info(
        f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B"
    )
    return model, tokenizer


def get_hidden_state(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    text: str,
    layers: List[int],
) -> Dict[int, np.ndarray]:
    """
    Returns mean-pooled hidden state at each requested layer.
    Shape per layer: (hidden_dim,)
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    ).to(model.device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # outputs.hidden_states: tuple[n_layers+1] of (1, seq_len, hidden_dim)
    result = {}
    for layer in layers:
        h = outputs.hidden_states[layer]   # (1, seq_len, dim)
        vec = h[0].mean(dim=0).cpu().float().numpy()  # (dim,)
        result[layer] = vec

    del outputs
    torch.cuda.empty_cache()

    return result


def extract_all(
    model,
    tokenizer,
    data: Dict[str, List[str]],
    layers: List[int],
    save_dir: str = "results/activations",
) -> Dict[str, Dict[int, List[np.ndarray]]]:
    """
    Extracts hidden states for all conditions.

    Returns:
        {
          "A": {8: [vec], 16: [vec], 24: [vec]},
          "B": {8: [vec, vec, ...], 16: [...], 24: [...]},
          "C": {8: [...], 16: [...], 24: [...]},
          "D": {8: [vec], 16: [vec], 24: [vec]},
        }
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    activations = {
        cond: {layer: [] for layer in layers}
        for cond in data.keys()
    }

    for cond, texts in data.items():
        logger.info(f"Extracting activations: Condition {cond} ({len(texts)} texts)")
        for i, text in enumerate(tqdm(texts, desc=f"Condition {cond}")):
            states = get_hidden_state(model, tokenizer, text, layers)
            for layer in layers:
                activations[cond][layer].append(states[layer])

            # Save each vector immediately (safe against crashes)
            for layer in layers:
                np.save(
                    f"{save_dir}/{cond}_{i:02d}_layer{layer}.npy",
                    states[layer]
                )

    logger.info(f"Activations saved to {save_dir}/")
    return activations


def load_activations_from_disk(
    save_dir: str,
    conditions: List[str],
    layers: List[int],
    n_per_condition: Dict[str, int],
) -> Dict[str, Dict[int, List[np.ndarray]]]:
    """Reload saved activations without re-running the model."""
    activations = {
        cond: {layer: [] for layer in layers}
        for cond in conditions
    }
    for cond in conditions:
        for i in range(n_per_condition[cond]):
            for layer in layers:
                path = Path(save_dir) / f"{cond}_{i:02d}_layer{layer}.npy"
                activations[cond][layer].append(np.load(path))
    logger.info("Activations reloaded from disk.")
    return activations
