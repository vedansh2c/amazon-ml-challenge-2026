# Progress

## Current state

- The repository contains reusable ML code, notebooks, scripts, local data, and outputs.
- The challenge statement and raw train/test TSVs are available.
- `notebooks/01_eda.ipynb` contains an unrun EDA workflow.
- A first disk-backed blocking, LightGBM matching, validation, and submission workflow is implemented under `src/pipelines/`.
- Shared dependencies are defined in the root `requirements.txt` for all experiments.
- Model runs use the config-driven `scripts/run_experiment.py` entry point. Blocking comparisons are explored in `notebooks/02_blocking_experiments.ipynb`.
- Reusable blocking code supports chunked character TF-IDF retrieval, a K sweep, persisted Parquet candidates, and missed-link reports. Its starting settings are in `configs/blocking_char_tfidf.json`.
- The per-K comparison output now reports only K, candidate recall, full-hit rate, average candidates, and missed true links; detailed calculations remain available through an opt-in formatter.
- Notebook blocking runs save timestamped artifacts under `outputs/experiments/<experiment_name>/`, including the exact config and compact metrics CSV.
- `docs/decisions.md` is the lightweight log template for future evidence-based experiment decisions.

## Decisions

- Keep challenge requirements in `docs/problem.md`.
- Keep reusable logic under `src/` and experiment-specific settings under `configs/`.
- Keep local datasets and generated outputs out of Git.
- Measure candidate recall on the complete training S2/S3 pool and tune the match threshold on held-out S1 entities using macro F0.5.

## Next steps

1. Use the blocking notebook to compare candidate-generation approaches and inspect missed links.
2. Select a blocker based on recall and candidate volume, then integrate it into the model pipeline.
3. Preserve the selected configuration and its measured artifacts for reproducibility.
