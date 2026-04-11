# Ablation Report (RunPod, 2026-04-11)

## Scope
- Ablation 1: structural confound via `C_hybrid` (same C agents, Yar JSON command schema).
- Ablation 2: length/control via `D_random` (5 random sentences from A) vs `D_distilled`.

## Artifacts
- Llama JSON: `/Users/vv/Downloads/AI/experiment2/ablation_results_runpod_20260411/llama_yar_attractor_v1_20260411_195752/results/yar_attractor_v1_20260411_195752.json`
- Gemma JSON: `/Users/vv/Downloads/AI/experiment2/ablation_results_runpod_20260411/gemma_yar_attractor_v1_20260411_195841/results/yar_attractor_v1_20260411_195841.json`

## Llama 3.1 8B
| Layer | mean(A+B→C) | mean(A+B→C_hybrid) | Δ(C_hybrid−C) | D_distilled | D_random | Δ(D_random−D_distilled) |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 0.026034 | 0.025851 | -0.000183 | 0.248277 | 0.197574 | -0.050703 |
| 16 | 0.032861 | 0.031289 | -0.001572 | 0.136044 | 0.120022 | -0.016022 |
| 24 | 0.022077 | 0.021056 | -0.001021 | 0.069443 | 0.057001 | -0.012442 |
| mean | - | - | -0.000925 | - | - | -0.026389 |

## Gemma 2 9B IT
| Layer | mean(A+B→C) | mean(A+B→C_hybrid) | Δ(C_hybrid−C) | D_distilled | D_random | Δ(D_random−D_distilled) |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 0.010733 | 0.009863 | -0.000870 | 0.058219 | 0.048598 | -0.009621 |
| 16 | 0.008164 | 0.008002 | -0.000162 | 0.039239 | 0.037760 | -0.001480 |
| 24 | 0.007478 | 0.007296 | -0.000182 | 0.026725 | 0.035980 | +0.009255 |
| mean | - | - | -0.000405 | - | - | -0.000615 |

## Answers to Research Questions
1. `C_hybrid ≈ C`?

- Yes. Across both models and all layers, `Δ(C_hybrid−C)` is small and negative (from −0.00157 to −0.00016 in magnitude range).
- Interpretation: in this setup, replacing command schema with Yar-style JSON does not increase distance from A+B; structural confound appears minimal.

2. `D_random ≈ D_distilled`?

- Mixed. Llama: `D_random` is lower on all layers. Gemma: lower on layers 8/16, but higher on layer 24.
- Interpretation: length-only account is insufficient; behavior depends on model/layer.

3. `D_random > D_distilled`?

- Not supported as a stable cross-model result.
- Only observed on Gemma layer 24 (`+0.009255`), while Llama shows the opposite on all layers.

## Practical Takeaway
- Ablation-1 supports robustness against the tested structural command confound.
- Ablation-2 does not yield a consistent superiority of distilled semantic summary over random 5-sentence slice across both models.
- Next recommended step: matched-length, multi-sample `D_random` bootstrap (e.g., 20-50 random slices) per model for stable inference.
