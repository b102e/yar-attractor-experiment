"""
Experiment: Identity as Attractor
Hypothesis: cognitive_core induces stable, specific region in LLM activation space
"""

from dataclasses import dataclass, field
from typing import List

@dataclass
class Config:
    # Model
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    pooling: str = "mean"
    layers: List[int] = field(default_factory=lambda: [8, 16, 24])
    conditions: List[str] = field(default_factory=lambda: ["A", "B", "C_prime"])

    # Reproducibility
    seed: int = 42

    # Paths
    data_dir: str = "data"
    results_dir: str = "results"
    figures_dir: str = "figures"
    activations_dir: str = "results/activations"

    # Experiment
    experiment_name: str = "yar_attractor_cprime"

    # Statistics
    n_bootstrap: int = 10000
    alpha: float = 0.05               # per-test
    alpha_bonferroni: float = 0.0167  # 0.05 / 3 layers

    # Baseline A+B x C distances from the main experiment JSON
    baseline_results_llama: str = "../results/llama/yar_attractor_v1_20260411_152017.json"
    baseline_results_gemma: str = "../results/gemma/yar_attractor_v1_20260411_160259.json"

    # t-SNE
    tsne_perplexity: int = 5
    tsne_random_state: int = 42

CONFIG = Config()
