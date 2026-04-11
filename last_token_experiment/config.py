"""
Experiment: Identity as Attractor
Hypothesis: cognitive_core induces stable, specific region in LLM activation space
"""

from dataclasses import dataclass, field
from typing import List

@dataclass
class Config:
    # Model
    model_name: str = "google/gemma-2-9b-it"
    pooling: str = "last"  # "mean" or "last"
    layers: List[int] = field(default_factory=lambda: [8, 16, 24])
    conditions: List[str] = field(
        default_factory=lambda: ["A", "B", "C", "C_hybrid", "D", "D_random"]
    )

    # Reproducibility
    seed: int = 42

    # Paths
    data_dir: str = "data"
    results_dir: str = "results"
    figures_dir: str = "figures"
    activations_dir: str = "results/activations"

    # Experiment
    experiment_name: str = "yar_attractor_v1"

    # Statistics
    n_bootstrap: int = 1000
    alpha: float = 0.05               # per-test
    alpha_bonferroni: float = 0.0167  # 0.05 / 3 layers

    # t-SNE
    tsne_perplexity: int = 5
    tsne_random_state: int = 42

CONFIG = Config()
