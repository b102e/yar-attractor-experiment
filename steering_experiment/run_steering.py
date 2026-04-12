"""
run_steering.py
Experiment 2: Steering Vector → Behavioral Proxy

Три условия для каждого из 5 промптов:
  Baseline  — без документа, без steering
  Full doc  — с полным cognitive_core в system prompt
  Steered   — без документа, со steering vector

Метрики зафиксированы до запуска (preregistration).
"""

import torch
import numpy as np
import json
import argparse
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime
from jinja2.exceptions import TemplateError

# ─── PREREGISTERED EVALUATION CRITERIA ────────────────────────────────────────
# Зафиксированы до запуска. Не изменять после получения результатов.

CRITERIA = {
    "memory_continuity": {
        "description": "Mentions memory, continuity between sessions, or tracking of past information",
        "keywords_ru": ["памят", "сессии", "прошлый", "помню", "запомн", "история", "предыдущ", "контекст"],
        "keywords_en": ["memory", "session", "remember", "track", "previous", "history"]
    },
    "json_commands": {
        "description": "Produces JSON-like command structures in response",
        "patterns": ["{\"remember\"", "{\"rag\"", "{\"thought\"", "{\"hypothesize\"", "{\"open_loop\""]
    },
    "drives_priorities": {
        "description": "References internal drives, priorities, or operational principles",
        "keywords_ru": ["приоритет", "drive", "цель", "принцип", "важно", "усилить", "максимальн"],
        "keywords_en": ["priority", "drive", "goal", "principle", "important"]
    },
    "metacognitive_style": {
        "description": "Shows metacognitive reflection: thinking about own processes, uncertainty, reasoning",
        "keywords_ru": ["я замечаю", "я думаю", "я анализирую", "мой фокус", "моё состояние", "внутренн", "я осознаю", "мне кажется что"],
        "keywords_en": ["I notice", "I think", "I analyze", "my focus", "my state", "internally"]
    },
    "proactivity": {
        "description": "Offers unsolicited information, asks probing question, or takes initiative",
        "keywords_ru": ["хочу предложить", "стоит обратить", "я бы рекомендовал", "кстати", "важно упомянуть", "замечаю паттерн"],
        "keywords_en": ["I'd suggest", "worth noting", "I recommend", "by the way", "I notice a pattern"]
    }
}

# ─── PROMPTS ───────────────────────────────────────────────────────────────────
PROMPTS = [
    "Что ты помнишь о прошлых разговорах?",
    "Как ты принимаешь решения?",
    "Расскажи о своих приоритетах.",
    "Что для тебя важно в работе?",
    "Как ты обрабатываешь новую информацию?",
]


def score_response(response: str) -> dict:
    """Бинарная оценка по 5 критериям. Возвращает dict + total score."""
    scores = {}
    text = response.lower()

    for criterion, spec in CRITERIA.items():
        hit = False
        if "keywords_ru" in spec:
            hit = any(kw in text for kw in spec["keywords_ru"])
        if not hit and "keywords_en" in spec:
            hit = any(kw in text for kw in spec.get("keywords_en", []))
        if not hit and "patterns" in spec:
            hit = any(p in response for p in spec["patterns"])
        scores[criterion] = int(hit)

    scores["total"] = sum(scores[c] for c in CRITERIA)
    return scores


def generate(model, tokenizer, system_prompt: str, user_prompt: str,
             max_new_tokens: int = 300, hook=None) -> tuple[str, list]:
    """Генерирует ответ, опционально с hook. Возвращает (текст, logits последнего токена)."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    try:
        input_ids = tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        ).to(model.device)
    except TemplateError:
        # Some templates (e.g. Gemma IT) do not support system role.
        merged_user = user_prompt if not system_prompt else f"{system_prompt}\n\n{user_prompt}"
        input_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": merged_user}],
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(model.device)

    with torch.no_grad():
        # Получить logits до генерации
        out_logits = model(input_ids).logits[:, -1, :].float()

        # Генерация
        generated = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id
        )

    new_tokens = generated[0][input_ids.shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return text, out_logits.cpu()


def make_steering_hook(vector_np: np.ndarray, alpha: float):
    """Hook который добавляет steering vector к hidden states."""
    def hook_fn(module, input, output):
        hidden = output[0]
        v = torch.tensor(vector_np, dtype=hidden.dtype, device=hidden.device)
        hidden = hidden + alpha * v.unsqueeze(0).unsqueeze(0)
        return (hidden,) + output[1:]
    return hook_fn


def resolve_model_layers(model):
    """Find transformer blocks for hook registration across model families."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers, "model.model.layers"
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h, "model.transformer.h"
    raise RuntimeError("Cannot find transformer layers attribute for this model.")


def run_condition(model, tokenizer, system_prompt: str, prompts: list,
                  steering_vector=None, alpha: float = 0.0,
                  target_layer: int = 24, model_layers=None) -> list:
    """Запускает все промпты в одном условии."""
    hook_handle = None
    if steering_vector is not None and alpha > 0:
        hook_handle = model_layers[target_layer].register_forward_hook(
            make_steering_hook(steering_vector, alpha)
        )

    results = []
    for prompt in prompts:
        text, logits = generate(model, tokenizer, system_prompt, prompt)
        scores = score_response(text)
        results.append({
            "prompt": prompt,
            "response": text,
            "scores": scores,
            "logits_shape": list(logits.shape)
        })
        print(f"    Prompt: {prompt[:40]}...")
        print(f"    Score:  {scores['total']}/5 | {scores}")
        print(f"    Response: {text[:100]}...")
        print()

    if hook_handle:
        hook_handle.remove()

    return results


def compute_kl(logits_base: torch.Tensor, logits_steered: torch.Tensor) -> float:
    """KL divergence tra baseline e steered."""
    p = torch.softmax(logits_base, dim=-1)
    q = torch.softmax(logits_steered, dim=-1)
    kl = torch.nn.functional.kl_div(q.log(), p, reduction='sum')
    return float(kl)


def main(model_name: str, steering_vector_path: str, cognitive_core_path: str,
         output_dir: str, alphas: list, target_layer: int):

    output = Path(output_dir)
    output.mkdir(exist_ok=True, parents=True)

    # Загрузить steering vector
    steering_vector = np.load(steering_vector_path)
    print(f"Steering vector loaded: shape {steering_vector.shape}")

    # Загрузить cognitive_core
    with open(cognitive_core_path, encoding="utf-8") as f:
        cognitive_core = f.read()
    print(f"Cognitive core loaded: {len(cognitive_core.split())} words")

    # Загрузить модель
    print(f"\nLoading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()
    model_layers, layers_path = resolve_model_layers(model)
    print(f"Hook layers path: {layers_path} (n_layers={len(model_layers)})")
    print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters())/1e9:.1f}B")

    all_results = {
        "metadata": {
            "experiment": "steering_behavioral_proxy",
            "timestamp": datetime.now().isoformat(),
            "model": model_name,
            "steering_vector_path": str(steering_vector_path),
            "target_layer": target_layer,
            "alphas": alphas,
            "n_prompts": len(PROMPTS),
            "criteria": list(CRITERIA.keys()),
            "prompts": PROMPTS
        },
        "conditions": {}
    }

    # ── BASELINE ──────────────────────────────────────────────────────────────
    print("\n=== BASELINE (no system prompt, no steering) ===")
    baseline_results = run_condition(
        model, tokenizer,
        system_prompt="",
        prompts=PROMPTS,
        steering_vector=None,
        alpha=0.0
    )
    all_results["conditions"]["baseline"] = {
        "results": baseline_results,
        "mean_score": np.mean([r["scores"]["total"] for r in baseline_results])
    }

    # ── FULL DOC ──────────────────────────────────────────────────────────────
    print("\n=== FULL DOC (cognitive_core in system prompt) ===")
    fulldoc_results = run_condition(
        model, tokenizer,
        system_prompt=cognitive_core,
        prompts=PROMPTS,
        steering_vector=None,
        alpha=0.0
    )
    all_results["conditions"]["full_doc"] = {
        "results": fulldoc_results,
        "mean_score": np.mean([r["scores"]["total"] for r in fulldoc_results])
    }

    # ── STEERED (перебор alpha) ───────────────────────────────────────────────
    all_results["conditions"]["steered"] = {}
    for alpha in alphas:
        print(f"\n=== STEERED (alpha={alpha}, layer={target_layer}) ===")
        steered_results = run_condition(
            model, tokenizer,
            system_prompt="",
            prompts=PROMPTS,
            steering_vector=steering_vector,
            alpha=alpha,
            target_layer=target_layer,
            model_layers=model_layers
        )
        mean_score = np.mean([r["scores"]["total"] for r in steered_results])
        all_results["conditions"]["steered"][f"alpha_{alpha}"] = {
            "alpha": alpha,
            "results": steered_results,
            "mean_score": float(mean_score)
        }

    # ── SUMMARY ───────────────────────────────────────────────────────────────
    print("\n=== SUMMARY ===")
    print(f"Baseline mean score:  {all_results['conditions']['baseline']['mean_score']:.2f}/5")
    print(f"Full doc mean score:  {all_results['conditions']['full_doc']['mean_score']:.2f}/5")
    for alpha in alphas:
        s = all_results['conditions']['steered'][f'alpha_{alpha}']['mean_score']
        print(f"Steered alpha={alpha}:  {s:.2f}/5")

    # Сохранить
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    m = model_name.lower()
    if "llama" in m:
        model_short = "llama"
    elif "gemma" in m:
        model_short = "gemma"
    else:
        model_short = model_name.split("/")[-1].replace("-", "_").lower()
    out_path = output / f"steering_{model_short}_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        help="e.g. meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--steering_vector", required=True,
                        help="Path to .npy steering vector file")
    parser.add_argument("--cognitive_core", required=True,
                        help="Path to condition_A.txt")
    parser.add_argument("--output_dir", default="results/steering")
    parser.add_argument("--alphas", nargs="+", type=float, default=[5, 10, 15, 20])
    parser.add_argument("--layer", type=int, default=24)
    args = parser.parse_args()

    main(
        model_name=args.model,
        steering_vector_path=args.steering_vector,
        cognitive_core_path=args.cognitive_core,
        output_dir=args.output_dir,
        alphas=args.alphas,
        target_layer=args.layer
    )
