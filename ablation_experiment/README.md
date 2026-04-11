# Ablation Experiment (Step 2/3)

This folder isolates the second experiment setup so it does not modify the original experiment at repo root.

## Purpose
- Step 2: structural-confound ablation with `condition_C_hybrid`.
- Step 3: distilled-length ablation with `condition_D_random`.

## What's included
- Dedicated experiment code snapshot:
  - `run.py`, `config.py`, `data_loader.py`, `compute_distances.py`
  - `extract_activations.py`, `visualize.py`, `verify_tokens.py`, `requirements.txt`
- Dedicated data snapshot in `data/`, including:
  - `condition_C_hybrid/C1_hybrid.txt ... C7_hybrid.txt`
  - `condition_D_random.txt`
  - plus original A/B/C/D files required to run the pipeline end-to-end.

## Key code differences vs original run
- Added loading/validation for `C_hybrid` and `D_random`.
- Added distance comparisons:
  - `A+B -> C` vs `A+B -> C_hybrid`
  - `D_distilled -> centroid(A+B)` vs `D_random -> centroid(A+B)`
- Added ablation fields in output JSON (`ablation_summary`) and extended console summary.

## Run
```bash
cd ablation_experiment
python run.py --dry-run
python run.py
```

## Notes
- This folder is intentionally separate to avoid mixing with the first experiment at repository root.
