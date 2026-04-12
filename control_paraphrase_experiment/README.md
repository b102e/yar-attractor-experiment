# Experiment 3 — Control Agent Paraphrases

Tests whether tight clustering is YAR-specific or a general property of semantically coherent documents with paraphrases.

## Data
- `data/condition_C1_original.txt` (copied from `data/condition_C/C1.txt`)
- `data/condition_C1_paraphrases/Sigma_B1..B7.txt`

## Run
```bash
python run.py --model meta-llama/Llama-3.1-8B-Instruct
python run.py --model google/gemma-2-9b-it
```

## Outputs
- `results/control_paraphrase/llama/control_paraphrase_*.json`
- `results/control_paraphrase/gemma/control_paraphrase_*.json`
- `experiment.log` per model
