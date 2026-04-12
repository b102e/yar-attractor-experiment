from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    pooling: str = "mean"
    layers: List[int] = field(default_factory=lambda: [8, 16, 24])
    seed: int = 42

    # Data
    yar_data_dir: str = "../data"
    sigma_data_dir: str = "data"

    # Output
    results_dir: str = "results/control_paraphrase"

    experiment_name: str = "control_agent_paraphrases"

    baseline_results_llama: str = "../results/llama/yar_attractor_v1_20260411_152017.json"
    baseline_results_gemma: str = "../results/gemma/yar_attractor_v1_20260411_160259.json"


CONFIG = Config()
