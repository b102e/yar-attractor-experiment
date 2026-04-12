"""
verify_tokens.py
Checks token lengths for C' files against Condition A (±15%).
"""

from pathlib import Path
from transformers import AutoTokenizer

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
DATA_DIR = Path("data")


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    text_a = (DATA_DIR / "condition_A.txt").read_text(encoding="utf-8")
    tokens_a = len(tokenizer.encode(text_a))
    min_tokens = int(tokens_a * 0.85)
    max_tokens = int(tokens_a * 1.15)

    print(f"Condition A: {tokens_a} tokens")
    print(f"Allowed range: {min_tokens}..{max_tokens}")

    folder = DATA_DIR / "condition_C_prime"
    all_ok = True
    for p in sorted(folder.glob("*.txt")):
        t = p.read_text(encoding="utf-8")
        n = len(tokenizer.encode(t))
        ok = min_tokens <= n <= max_tokens
        all_ok = all_ok and ok
        print(f"{p.name}: {n} -> {'OK' if ok else 'FAIL'}")

    print("PASS" if all_ok else "FAIL")


if __name__ == "__main__":
    main()
