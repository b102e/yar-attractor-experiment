# D_random Bootstrap Results

Multi-sample bootstrap comparison of distilled D vs 30 random 5-sentence samples from A.

## Outputs
- `results/bootstrap/llama/bootstrap_llama.json`
- `results/bootstrap/gemma/bootstrap_gemma.json`

## Key Result
For both models (Llama 3.1 8B Instruct and Gemma 2 9B IT),
`distilled_beats_random_pct = 100%` on layers 8, 16, 24.

Interpretation: the D_distilled vector is consistently closer to A+B centroid than random same-length fragments, indicating a semantic effect rather than pure length effect.
