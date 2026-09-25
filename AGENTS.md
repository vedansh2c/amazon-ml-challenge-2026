# Repository guidance

## Project context

- Read [docs/problem.md](docs/problem.md) before making challenge-specific choices.
- Check [docs/progress.md](docs/progress.md) before starting substantial work.
- The raw challenge data is under `data/raw/student_resource/dataset/`.
- Source files are tab-separated and use `entity_id`, `business_name`, `business_address`, and `country`.
- Source 1 is the reference source; Source 2 and Source 3 contain possible matches.
- Country labels are open-ended. Do not hard-code the analysis or pipeline to India and US; the test set includes France.
- External business lookup, geocoding, internet enrichment, and external entity-resolution services are prohibited.

## Code and data organization

- Keep reusable data, feature, model, training, and pipeline code under `src/`.
- Keep experiment-specific choices in JSON/config files under `configs/` or `experiments/`.
- Use `scripts/run_experiment.py` as the config-driven experiment entry point. Do not create a new runner script for every parameter variation.
- Set the experiment `device` to `auto` for GPU-when-available with CPU fallback; use `cpu` or `gpu` to force a choice. 
- Use the shared root `requirements.txt`; do not create one dependency file per experiment unless an experiment truly requires an isolated environment.
- Put local datasets under `data/` and generated artifacts under `outputs/`. Do not modify raw datasets.
- Keep notebooks for exploration and reproducible analysis; move reusable logic discovered there into `src/` when appropriate.

## Execution and validation

- Do not run full training, large indexing jobs, full inference, or large data scans unless the user explicitly asks for execution.
- By default, perform lightweight checks such as syntax compilation, config validation, and code review.
- When execution is explicitly requested, report the command, configuration, runtime-relevant assumptions, and measured result. Never claim a run completed without evidence.
- Submission files must be tab-separated and contain every test Source 1 entity exactly once. Matches may reference only existing Source 2 or Source 3 IDs, with no duplicates.
- Validate generated submissions with `data/raw/student_resource/utils/validate_submission.py`.

## Documentation and reproducibility

- After a meaningful code or configuration change, update `docs/progress.md` with what changed and the next step.
- The user maintains experiment results; do not invent, overwrite, or claim experiment metrics without a recorded run.
- Use fixed random seeds for sampled analyses or experiments when randomness is involved.
- Do not invent challenge requirements or conclusions unsupported by the available data.
