"""
visualize.py
Generates all figures for the preprint.
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
from typing import Dict, List

from sklearn.manifold import TSNE

logger = logging.getLogger(__name__)

# Publication style
mpl.rcParams.update({
    "font.family":    "DejaVu Sans",
    "font.size":      12,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "figure.dpi":     150,
    "axes.grid":      True,
    "grid.alpha":     0.3,
    "grid.linestyle": "--",
})

COLORS = {
    "A": "#2166ac",   # dark blue  — original
    "B": "#74add1",   # light blue — paraphrases
    "C": "#d73027",   # red        — control
    "D": "#1a9641",   # green      — distilled
}

LABELS = {
    "A": "Original (A)",
    "B": "Paraphrases (B)",
    "C": "Control (C)",
    "D": "Distilled (D)",
}


def _save(name: str, figures_dir: str):
    Path(figures_dir).mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/{name}.png", dpi=150, bbox_inches="tight")
    plt.savefig(f"{figures_dir}/{name}.pdf", bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {figures_dir}/{name}.png")


# ── Figure 1: t-SNE per layer ─────────────────────────────────────────────────

def plot_tsne(
    activations: Dict[str, Dict[int, List[np.ndarray]]],
    layers: List[int],
    figures_dir: str = "figures",
    perplexity: int = 5,
    random_state: int = 42,
):
    fig, axes = plt.subplots(1, len(layers), figsize=(5 * len(layers), 4.5))
    if len(layers) == 1:
        axes = [axes]

    for ax, layer in zip(axes, layers):
        all_vecs, all_labels = [], []
        for cond in ["A", "B", "C", "D"]:
            for vec in activations[cond][layer]:
                all_vecs.append(vec)
                all_labels.append(cond)

        X = np.stack(all_vecs)
        tsne = TSNE(
            n_components=2,
            perplexity=min(perplexity, len(all_vecs) - 1),
            random_state=random_state,
        )
        X2d = tsne.fit_transform(X)

        offset = 0
        for cond in ["C", "B", "A", "D"]:   # draw C first (background)
            n = len(activations[cond][layer])
            idx = [i for i, l in enumerate(all_labels) if l == cond]
            marker = "o" if cond != "D" else "*"
            size   = 80 if cond != "D" else 200
            ax.scatter(
                X2d[idx, 0], X2d[idx, 1],
                c=COLORS[cond], label=LABELS[cond],
                marker=marker, s=size, alpha=0.85,
                edgecolors="white", linewidths=0.5,
                zorder=3 if cond in ("A", "D") else 2,
            )

        ax.set_title(f"Layer {layer}")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.legend(fontsize=9, loc="best")

    fig.suptitle(
        "Hidden state representations by condition across layers",
        fontsize=13, y=1.02,
    )
    _save("fig1_tsne", figures_dir)


# ── Figure 2: Convergence curve ───────────────────────────────────────────────

def plot_convergence(
    distance_results: Dict[int, Dict],
    layers: List[int],
    figures_dir: str = "figures",
):
    within_means = [distance_results[l]["stats"]["mean_within_AB"] for l in layers]
    within_stds  = [distance_results[l]["stats"]["std_within_AB"]  for l in layers]
    between_means = [distance_results[l]["stats"]["mean_between"]  for l in layers]
    between_stds  = [distance_results[l]["stats"]["std_between"]   for l in layers]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.errorbar(
        layers, within_means, yerr=within_stds,
        marker="o", color=COLORS["A"], label="Within A+B (paraphrases)",
        linewidth=2, capsize=4,
    )
    ax.errorbar(
        layers, between_means, yerr=between_stds,
        marker="s", color=COLORS["C"], label="Between A+B vs C (control)",
        linewidth=2, capsize=4, linestyle="--",
    )

    # Significance markers
    for layer in layers:
        if distance_results[layer]["stats"]["significant"]:
            y_max = max(
                distance_results[layer]["stats"]["mean_within_AB"],
                distance_results[layer]["stats"]["mean_between"],
            )
            ax.annotate(
                "*", xy=(layer, y_max + 0.005),
                ha="center", fontsize=14, color="black",
            )

    ax.set_xlabel("Layer index")
    ax.set_ylabel("Mean cosine distance")
    ax.set_title("Representation convergence across layers\n(* = p < 0.0167, Bonferroni)")
    ax.legend()
    ax.set_xticks(layers)

    _save("fig2_convergence", figures_dir)


# ── Figure 3: Distance matrix heatmap at layer 16 ────────────────────────────

def plot_distance_matrix(
    activations: Dict[str, Dict[int, List[np.ndarray]]],
    layer: int = 16,
    figures_dir: str = "figures",
):
    from sklearn.metrics.pairwise import cosine_distances

    all_vecs, tick_labels, tick_colors = [], [], []
    for cond in ["A", "B", "C", "D"]:
        for i, vec in enumerate(activations[cond][layer]):
            all_vecs.append(vec)
            tick_labels.append(f"{cond}{i+1}" if cond != "A" else "A")
            tick_colors.append(COLORS[cond])

    D = cosine_distances(np.stack(all_vecs))

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(D, cmap="coolwarm", vmin=0, vmax=D.max())
    plt.colorbar(im, ax=ax, label="Cosine distance")

    ax.set_xticks(range(len(tick_labels)))
    ax.set_yticks(range(len(tick_labels)))
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(tick_labels, fontsize=9)

    # Color tick labels by condition
    for tick, color in zip(ax.get_xticklabels(), tick_colors):
        tick.set_color(color)
    for tick, color in zip(ax.get_yticklabels(), tick_colors):
        tick.set_color(color)

    ax.set_title(f"Pairwise cosine distance matrix (layer {layer})")

    _save(f"fig3_distance_matrix_layer{layer}", figures_dir)


# ── Figure 4: D distance to AB centroid across layers ────────────────────────

def plot_distilled_trajectory(
    distance_results: Dict[int, Dict],
    layers: List[int],
    figures_dir: str = "figures",
):
    d_dists = [distance_results[l]["d_to_centroid_AB"] for l in layers]
    within  = [distance_results[l]["stats"]["mean_within_AB"] for l in layers]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(layers, d_dists, marker="*", color=COLORS["D"],
            linewidth=2, markersize=12, label="Distilled (D) → centroid A+B")
    ax.plot(layers, within, marker="o", color=COLORS["A"],
            linewidth=2, linestyle="--", label="Mean within A+B")

    ax.set_xlabel("Layer index")
    ax.set_ylabel("Cosine distance")
    ax.set_title("Distilled cognitive_core vs. full paraphrase cluster\n(exploratory H3)")
    ax.legend()
    ax.set_xticks(layers)

    _save("fig4_distilled_trajectory", figures_dir)


# ── Generate all figures ──────────────────────────────────────────────────────

def generate_all(
    activations: Dict,
    distance_results: Dict,
    layers: List[int],
    figures_dir: str = "figures",
    tsne_perplexity: int = 5,
):
    logger.info("Generating figures...")
    plot_tsne(activations, layers, figures_dir, perplexity=tsne_perplexity)
    plot_convergence(distance_results, layers, figures_dir)
    plot_distance_matrix(activations, layer=16, figures_dir=figures_dir)
    plot_distilled_trajectory(distance_results, layers, figures_dir)
    logger.info(f"All figures saved to {figures_dir}/")
