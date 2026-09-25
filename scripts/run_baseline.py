#!/usr/bin/env python3
"""Train, validate, and generate the first entity-resolution submission."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipelines.baseline import predict, train  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "predict", "all"), default="all")
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/raw/student_resource/dataset")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/baseline")
    parser.add_argument("--sample-divisor", type=int, default=100,
                        help="Select approximately 1/N training S1 records; use 1 for all")
    parser.add_argument("--per-key", type=int, default=30)
    parser.add_argument("--final-limit", type=int, default=50)
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel processes for full test inference")
    parser.add_argument("--device", choices=("auto", "cpu", "gpu"), default="auto")
    args = parser.parse_args()
    if min(args.sample_divisor, args.per_key, args.final_limit, args.workers) < 1:
        parser.error("sample-divisor, per-key, final-limit, and workers must be positive")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.mode in ("train", "all"):
        train(args.dataset, args.output, sample_divisor=args.sample_divisor,
              per_key=args.per_key, final_limit=args.final_limit, device=args.device)
    if args.mode in ("predict", "all"):
        predict(args.dataset, args.output, per_key=args.per_key,
                final_limit=args.final_limit, workers=args.workers)


if __name__ == "__main__":
    main()
