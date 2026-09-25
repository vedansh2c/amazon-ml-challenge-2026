#!/usr/bin/env python3
"""Run an experiment from a JSON configuration file."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipelines.baseline import predict, train  # noqa: E402


def path_from_config(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = json.loads(config_path.read_text())
    if config.get("pipeline") != "baseline":
        raise ValueError("This runner currently supports pipeline='baseline'. Add new pipeline dispatch here.")
    mode = config.get("mode", "all")
    if mode not in {"train", "predict", "all"}:
        raise ValueError("mode must be train, predict, or all")
    dataset = path_from_config(config.get("dataset", "data/raw/student_resource/dataset"))
    output = path_from_config(config.get("output", "outputs/experiment"))
    sample_divisor = int(config.get("sample_divisor", 100))
    per_key = int(config.get("per_key", 30))
    final_limit = int(config.get("final_limit", 50))
    workers = int(config.get("workers", 4))
    device = str(config.get("device", "auto")).lower()
    if min(sample_divisor, per_key, final_limit, workers) < 1:
        raise ValueError("numeric experiment settings must be positive")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if mode in {"train", "all"}:
        train(dataset, output, sample_divisor=sample_divisor,
              per_key=per_key, final_limit=final_limit, device=device)
    if mode in {"predict", "all"}:
        predict(dataset, output, per_key=per_key,
                final_limit=final_limit, workers=workers)


if __name__ == "__main__":
    main()
