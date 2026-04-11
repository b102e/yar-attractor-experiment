"""
d_random_bootstrap.py
Generates multiple D_random bootstrap samples from condition_A.txt.

Usage:
    python d_random_bootstrap.py
"""

from pathlib import Path
import random
import re


def split_sentences(text: str):
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]


def main():
    base = Path(__file__).resolve().parent
    src = base / "data" / "condition_A.txt"
    out_dir = base / "data" / "condition_D_random_bootstrap"
    out_dir.mkdir(parents=True, exist_ok=True)

    text = src.read_text(encoding="utf-8")
    sents = split_sentences(text)
    if len(sents) < 5:
        raise RuntimeError(f"Need at least 5 sentences in {src}, got {len(sents)}")

    rng = random.Random(42)
    for i in range(30):
        picked = rng.sample(sents, 5)
        out = out_dir / f"D_random_{i:02d}.txt"
        out.write_text(" ".join(picked) + "\n", encoding="utf-8")

    print(f"Generated 30 files in: {out_dir}")


if __name__ == "__main__":
    main()
