# Last-Token Pooling Experiment

Isolated code snapshot for the last-token pooling ablation.

## Goal
Compare hidden-state pooling strategies:
- `mean` pooling (baseline)
- `last` token pooling (this experiment)

## Key changes
- `config.py`: `pooling` field (`"mean"` or `"last"`)
- `extract_activations.py`: selectable pooling logic per layer
- `run.py`: CLI overrides
  - `--model <hf-model-id>`
  - `--pooling mean|last`

## Commands used
```bash
python run.py --model meta-llama/Llama-3.1-8B-Instruct --pooling last
python run.py --model google/gemma-2-9b-it --pooling last
```

## Output location in repository
See:
- `results/last_token/llama/`
- `results/last_token/gemma/`
