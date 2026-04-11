# Multi-sample D_random Bootstrap Experiment

This folder contains scripts for Task 2 bootstrap analysis (no GPU required for analysis stage).

## Scripts

- `d_random_bootstrap.py`
  - Generates 30 random 5-sentence files from `data/condition_A.txt`
  - Output: `data/condition_D_random_bootstrap/D_random_00.txt ... D_random_29.txt`

- `bootstrap_analysis.py`
  - Compares distances from `D_distilled` and `D_random_*` to A+B centroid
  - Input: activation `.npy` files
  - Output: `bootstrap_<model>.json`

## Run

```bash
python d_random_bootstrap.py
python bootstrap_analysis.py --activations_dir <path_to_activations> --model llama
python bootstrap_analysis.py --activations_dir <path_to_activations> --model gemma
```
