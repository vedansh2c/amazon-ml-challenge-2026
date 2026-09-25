# ML challenge

This repository is organized so experiments can share data preparation, feature engineering, model, and training code as the project grows.

Start with [the problem statement](docs/problem.md). [Progress](docs/progress.md) records the current state, and [AGENTS.md](AGENTS.md) gives coding agents the repository's working conventions.

```text
configs/         Shared configuration
docs/            Problem statement, progress, and experiment record
src/
  data/          Data loading, validation, and splitting
  features/      Reusable feature transformations
  models/        Model implementations
  training/      Training and evaluation logic
  pipelines/     Reusable workflows that connect the components
notebooks/       Exploration, EDA, and blocking experiments
scripts/         Config-driven model experiment entry point
data/            Local datasets (ignored by Git)
outputs/         Generated models, metrics, and predictions (ignored by Git)
```

Add reusable logic under `src/` and keep experiment-specific choices under `configs/`.

## Running experiments

All experiments use the shared environment in `requirements.txt`. Create it once:

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/run_experiment.py --config configs/baseline.json
```

Model runs use `scripts/run_experiment.py`; the JSON config selects the model pipeline and its settings. Blocking exploration happens in notebooks and uses reusable implementations from `src/`.

Set `"device": "auto"` to try GPU LightGBM and fall back to CPU when GPU support is unavailable. Use `"cpu"` to skip the GPU attempt or `"gpu"` to require GPU support. Indexing, blocking, and feature generation remain CPU-based.

The model experiment runner trains and validates on a reproducible S1 sample and saves the trained model under the configured output directory. It does not run test prediction. Keep each experiment's configuration and measured result alongside its generated artifacts under `outputs/`.

## Blocking experiments

Compare blocking approaches in [the blocking experiments notebook](notebooks/02_blocking_experiments.ipynb). It uses reusable code under `src/` and defaults to the settings in `configs/blocking_char_tfidf.json`. The blocker searches the complete training S2/S3 corpus in chunks. Its sample divisor controls how many Source 1 training records are evaluated; setting it to 1 evaluates them all and is substantially more expensive.

Results are saved under `outputs/experiments/<experiment_name>/<run_id>/`: `candidates.parquet` contains ranked candidate pairs for the largest K, `comparison.tsv` and `metrics.json` contain the K sweep, and `missed_pairs.tsv` lists missed true links at each K. The directory also contains the config snapshot and a manifest with data, code, and library signatures. Each run gets a timestamped directory so previous artifacts remain available.
