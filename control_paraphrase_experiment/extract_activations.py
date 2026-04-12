import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


def load_model(model_name: str, device: str = "auto") -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    logger.info("Loading model: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map=device)
    model.eval()
    return model, tokenizer


def get_hidden_state(model, tokenizer, text: str, layers: List[int], pooling: str = "mean") -> Dict[int, np.ndarray]:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    result = {}
    for layer in layers:
        h = outputs.hidden_states[layer][0]
        if pooling == "mean":
            vec = h.mean(dim=0)
        elif pooling == "last":
            vec = h[-1]
        else:
            raise ValueError(f"Unsupported pooling: {pooling}")
        result[layer] = vec.cpu().float().numpy()
    del outputs
    torch.cuda.empty_cache()
    return result


def extract_all(model, tokenizer, data: Dict[str, List[str]], layers: List[int], save_dir: str, pooling: str = "mean"):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    activations = {c: {l: [] for l in layers} for c in data.keys()}

    for cond, texts in data.items():
        logger.info("Extracting %s (%d texts)", cond, len(texts))
        for i, text in enumerate(tqdm(texts, desc=cond)):
            states = get_hidden_state(model, tokenizer, text, layers, pooling=pooling)
            for layer in layers:
                activations[cond][layer].append(states[layer])
                np.save(f"{save_dir}/{cond}_{i:02d}_layer{layer}.npy", states[layer])
    return activations


def load_activations_from_disk(save_dir: str, conditions: List[str], layers: List[int], n_per_condition: Dict[str, int]):
    out = {c: {l: [] for l in layers} for c in conditions}
    for cond in conditions:
        for i in range(n_per_condition[cond]):
            for layer in layers:
                p = Path(save_dir) / f"{cond}_{i:02d}_layer{layer}.npy"
                out[cond][layer].append(np.load(p))
    return out
