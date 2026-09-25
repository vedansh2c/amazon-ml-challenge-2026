# Progress

## Current state

- The repository has a folder scaffold for reusable ML code, experiments, notebooks, scripts, tests, local data, and outputs.
- The challenge statement and raw train/test TSVs are available.
- `notebooks/01_eda.ipynb` contains an unrun EDA workflow.
- A first disk-backed blocking, LightGBM matching, validation, and submission workflow is implemented in `src/` and `scripts/run_baseline.py`.
- Shared dependencies are defined in the root `requirements.txt` for all experiments.
- Experiments can be launched through the config-driven `scripts/run_experiment.py` runner.

## Decisions

- Keep challenge requirements in `docs/problem.md`.
- Keep reusable logic under `src/` and experiment-specific work under `experiments/` or `configs/`.
- Keep local datasets and generated outputs out of Git.
- Measure candidate recall on the complete training S2/S3 pool and tune the match threshold on held-out S1 entities using macro F0.5.

## Next steps

1. Run the baseline, inspect full-pool candidate recall and validation macro F0.5, and record the result in `docs/experiments.md`.
2. Validate the two test output files with the provided submission validator.
3. Use the measured blocking misses to decide the next retrieval improvement.
