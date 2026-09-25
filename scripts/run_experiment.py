#!/usr/bin/env python3
"""Train and validate the configured entity matching model."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = json.loads(config_path.read_text())
    if config.get("pipeline") != "baseline":
        parser.error("config pipeline must be 'baseline'")
    dataset_value = Path(config.get("dataset", "data/raw/student_resource/dataset")).expanduser()
    output_value = Path(config.get("output", "outputs/baseline")).expanduser()
    dataset = dataset_value if dataset_value.is_absolute() else ROOT / dataset_value
    output = output_value if output_value.is_absolute() else ROOT / output_value
    sample_divisor = int(config.get("sample_divisor", 100))
    per_key = int(config.get("per_key", 30))
    final_limit = int(config.get("final_limit", 50))
    device = str(config.get("device", "auto")).lower()
    if min(sample_divisor, per_key, final_limit) < 1:
        parser.error("sample_divisor, per_key, and final_limit must be positive")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from src.pipelines.baseline import train

    train(dataset, output, sample_divisor=sample_divisor, per_key=per_key,
          final_limit=final_limit, device=device)


if __name__ == "__main__":
    main()
