from pathlib import Path
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


def load_yar_A(yar_data_dir: str) -> str:
    p = Path(yar_data_dir) / "condition_A.txt"
    return p.read_text(encoding="utf-8").strip()


def load_yar_B(yar_data_dir: str) -> List[str]:
    folder = Path(yar_data_dir) / "condition_B"
    return [p.read_text(encoding="utf-8").strip() for p in sorted(folder.glob("*.txt"))]


def load_sigma_original(sigma_data_dir: str) -> str:
    p = Path(sigma_data_dir) / "condition_C1_original.txt"
    return p.read_text(encoding="utf-8").strip()


def load_sigma_paraphrases(sigma_data_dir: str) -> List[str]:
    folder = Path(sigma_data_dir) / "condition_C1_paraphrases"
    return [p.read_text(encoding="utf-8").strip() for p in sorted(folder.glob("*.txt"))]


def load_all(yar_data_dir: str, sigma_data_dir: str) -> Dict[str, List[str]]:
    data = {
        "YAR_A": [load_yar_A(yar_data_dir)],
        "YAR_B": load_yar_B(yar_data_dir),
        "SIGMA_ORIGINAL": [load_sigma_original(sigma_data_dir)],
        "SIGMA_B": load_sigma_paraphrases(sigma_data_dir),
    }
    logger.info("Loaded: YAR_A=%d YAR_B=%d SIGMA_ORIGINAL=%d SIGMA_B=%d",
                len(data["YAR_A"]), len(data["YAR_B"]), len(data["SIGMA_ORIGINAL"]), len(data["SIGMA_B"]))
    return data


def verify_data(data: Dict[str, List[str]]) -> bool:
    ok = True
    if len(data["YAR_A"]) != 1:
        logger.error("YAR_A must have exactly 1 file")
        ok = False
    if len(data["YAR_B"]) != 7:
        logger.error("YAR_B must have 7 files")
        ok = False
    if len(data["SIGMA_ORIGINAL"]) != 1:
        logger.error("SIGMA_ORIGINAL must have exactly 1 file")
        ok = False
    if len(data["SIGMA_B"]) != 7:
        logger.error("SIGMA_B must have 7 files")
        ok = False
    return ok
