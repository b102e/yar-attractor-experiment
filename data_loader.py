"""
data_loader.py
Loads condition texts from data/ directory.

Expected structure:
    data/
        condition_A.txt          # original cognitive_core
        condition_B/
            B1.txt ... B7.txt    # paraphrases
        condition_C/
            C1.txt ... C7.txt    # control prompts
        condition_D.txt          # distilled core
"""

from pathlib import Path
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


def load_condition_A(data_dir: str = "data") -> str:
    path = Path(data_dir) / "condition_A.txt"
    text = path.read_text(encoding="utf-8").strip()
    logger.info(f"Condition A loaded: {len(text.split())} words")
    return text


def load_condition_B(data_dir: str = "data") -> List[str]:
    folder = Path(data_dir) / "condition_B"
    texts = []
    for p in sorted(folder.glob("*.txt")):
        texts.append(p.read_text(encoding="utf-8").strip())
    logger.info(f"Condition B loaded: {len(texts)} paraphrases")
    return texts


def load_condition_C(data_dir: str = "data") -> List[str]:
    folder = Path(data_dir) / "condition_C"
    texts = []
    for p in sorted(folder.glob("*.txt")):
        texts.append(p.read_text(encoding="utf-8").strip())
    logger.info(f"Condition C loaded: {len(texts)} control prompts")
    return texts


def load_condition_D(data_dir: str = "data") -> str:
    path = Path(data_dir) / "condition_D.txt"
    text = path.read_text(encoding="utf-8").strip()
    logger.info(f"Condition D loaded: {len(text.split())} words")
    return text


def load_all(data_dir: str = "data") -> Dict:
    return {
        "A": [load_condition_A(data_dir)],
        "B": load_condition_B(data_dir),
        "C": load_condition_C(data_dir),
        "D": [load_condition_D(data_dir)],
    }


def verify_data(data: Dict) -> bool:
    """Basic sanity checks before running experiment."""
    ok = True

    if len(data["A"]) != 1:
        logger.error("Condition A must have exactly 1 document")
        ok = False

    if len(data["B"]) < 5:
        logger.error(f"Condition B has only {len(data['B'])} paraphrases, need at least 5")
        ok = False

    if len(data["C"]) < 5:
        logger.error(f"Condition C has only {len(data['C'])} controls, need at least 5")
        ok = False

    if len(data["D"]) != 1:
        logger.error("Condition D must have exactly 1 document")
        ok = False

    # Token length rough check (1 word ≈ 1.3 tokens)
    len_A = len(data["A"][0].split())
    for i, text in enumerate(data["C"]):
        len_C = len(text.split())
        ratio = len_C / len_A
        if ratio < 0.7 or ratio > 1.3:
            logger.warning(
                f"Condition C{i+1} length ratio vs A: {ratio:.2f} "
                f"(target 0.85–1.15)"
            )

    if ok:
        logger.info("Data verification passed.")
    return ok
