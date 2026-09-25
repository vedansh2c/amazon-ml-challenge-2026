# ML challenge

This repository is organized so experiments can share data preparation, feature engineering, model, and training code as the project grows.

Start with [the problem statement](docs/problem.md) when it is available. [Progress](docs/progress.md) records the current state, [experiments](docs/experiments.md) records completed runs, and [AGENTS.md](AGENTS.md) gives coding agents the repository's working conventions.

```text
configs/         Shared configuration
docs/            Problem statement, progress, and experiment record
src/
  data/          Data loading, validation, and splitting
  features/      Reusable feature transformations
  models/        Model implementations
  training/      Training and evaluation logic
  pipelines/     Reusable workflows that connect the components
experiments/     Experiment-specific configurations and code
notebooks/       Exploration and analysis
scripts/         Commands for training, evaluation, and prediction
tests/           Checks for reusable behavior
data/            Local datasets (ignored by Git)
outputs/         Generated models, metrics, and predictions (ignored by Git)
```

Add reusable logic under `src/` and keep experiment-specific choices under `experiments/` or `configs/`. The pipeline implementation can be added once the challenge's dataset and task are known.

## Running experiments

All experiments use the shared environment in `requirements.txt`. Create it once:

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/run_experiment.py --config configs/baseline.json
```

Each experiment should keep its reusable code under `src/`, its choices under `configs/` or `experiments/`, and its generated artifacts under a separate `outputs/<experiment_name>/` directory. To create another run, copy `configs/baseline.json`, change its values, and run the same generic command. No new runner script is needed.

Set `"device": "auto"` to try GPU LightGBM and fall back to CPU when GPU support is unavailable. Use `"cpu"` to skip the GPU attempt or `"gpu"` to require GPU support. Indexing, blocking, and feature generation remain CPU-based.

The baseline builds disk-backed indexes over the complete S2/S3 files, trains on a reproducible S1 sample, validates macro F0.5, and writes predictions. Its outputs are under `outputs/baseline/`. Validate its submission files with:

```sh
python3 data/raw/student_resource/utils/validate_submission.py \
  --matching outputs/baseline/submission/matching_results.tsv \
  --candidate outputs/baseline/submission/candidate_pairs.tsv \
  --test-dir data/raw/student_resource/dataset/test
```

Use `--mode train` or `--mode predict` to rerun one stage. New experiments should follow the same pattern and record their configuration and measured result in `docs/experiments.md`. The source index is reused when the corresponding raw file sizes and modification times match.
