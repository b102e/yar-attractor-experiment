# YAR Attractor Experiment

**Hypothesis:** The cognitive_core of a persistent cognitive agent behaves as a 
conceptual attractor in LLM activation space — analogous to semantic concept 
attractors reported by Chytas & Singh (2025).

## Setup

```bash
pip install -r requirements.txt
```

## Data structure

```
data/
    condition_A.txt          # original cognitive_core (full document)
    condition_B/
        B1.txt               # paraphrase 1
        B2.txt               # paraphrase 2
        ...
        B7.txt               # paraphrase 7
    condition_C/
        C1.txt               # control prompt 1
        ...
        C7.txt               # control prompt 7
    condition_D.txt          # distilled core (5 sentences)
```

## Run

```bash
# Full run (requires GPU, ~16GB VRAM for Llama 3.1 8B)
python run.py

# Verify data only
python run.py --dry-run

# Skip model, reload saved activations
python run.py --skip-extraction
```

## Output

```
results/
    yar_attractor_v1_YYYYMMDD_HHMMSS.json   # full results with stats
    activations/                              # raw .npy files per condition/layer
    experiment.log

figures/
    fig1_tsne.png / .pdf
    fig2_convergence.png / .pdf
    fig3_distance_matrix_layer16.png / .pdf
    fig4_distilled_trajectory.png / .pdf
```

## Interpreting results

Primary claim requires:
- `p_value < 0.0167` (Bonferroni α) at layer 16 and/or 24
- `mean_within_AB < mean_between`
- `cohens_d > 0` (positive effect size)

Convergence (H2): `mean_within_AB` should decrease from layer 8 → 24.

Distilled core (H3, exploratory): `d_to_centroid_AB` at layer 24 should be
comparable to `mean_within_AB` — indicating the 5-sentence distillation
reaches the same attractor as the full document.

## Reference

Chytas, S.P. & Singh, V. (2025). Concept Attractors in LLMs and their 
Applications. arXiv:2601.11575
