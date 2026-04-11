# YAR Attractor Experiment

This repository contains code and data for the paper:

**"Identity as Attractor: Geometric Evidence for Persistent Agent Architecture in LLM Activation Space"**  
Vladimir Vasilenko, 2026

---

## Hypothesis

The cognitive_core of a persistent cognitive agent induces attractor-like structure in LLM activation space, analogous to semantic concept clustering reported by Chytas & Singh (2025).

---

## Setup

```bash
pip install -r requirements.txt
```

## Data structure

```text
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

```text
results/
    yar_attractor_v1_YYYYMMDD_HHMMSS.json   # full results with stats
    activations/                            # raw .npy files per condition/layer
    experiment.log

figures/
    fig1_tsne.png / .pdf
    fig2_convergence.png / .pdf
    fig3_distance_matrix_layer16.png / .pdf
    fig4_distilled_trajectory.png / .pdf
```

## Interpreting results

Primary result (H1):
- p_value < 0.0167 (Bonferroni α) at layer 16 and/or 24
- mean_within_AB < mean_between
- cohens_d >> 0 (typically > 1.0 in observed results)

Convergence (H2):
- mean_within_AB decreases from layer 8 → 24 (allowing minor non-monotonicity)

Distilled core (H3, exploratory):
- d_to_centroid_AB decreases across layers
- but remains substantially larger than mean_within_AB
- → indicating partial convergence without reaching the full-document region

## Notes

- The experiment relies on mean pooling over token positions.
- Last-token pooling does not reproduce the effect (see paper).
- Structural markers (JSON blocks, delimiters) have a minor contribution relative to semantic content (see ablation results).

## Reproducibility

All results can be reproduced with a single command (see Run section).
Full code, data, and experiment configuration are included in this repository.

## References

Chytas, S.P. & Singh, V. (2025).  
Concept Attractors in LLMs and their Applications.  
arXiv:2601.11575

Lu, C. et al. (2026).  
The Assistant Axis: Situating and stabilizing the default persona of language models.  
arXiv:2601.10387
