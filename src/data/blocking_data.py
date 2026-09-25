"""Chunked TSV loading for blocking; raw text columns are preserved."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
TRUTH_COLUMNS = ["source1_entity_id", "matched_entity_ids"]


def source_path(dataset: Path, source: int) -> Path:
    if source not in (1, 2, 3):
        raise ValueError("source must be 1, 2, or 3")
    return dataset / "train" / f"train_source{source}.tsv"


def iter_source(path: Path, chunk_size: int):
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    reader = pd.read_csv(path, sep="\t", chunksize=chunk_size,
                         dtype="string", keep_default_na=False)
    for chunk in reader:
        if list(chunk.columns) != SOURCE_COLUMNS:
            raise ValueError(f"Unexpected source columns in {path}: {list(chunk.columns)}")
        yield chunk


def load_source1_sample(path: Path, *, divisor: int, chunk_size: int) -> tuple[pd.DataFrame, int]:
    """Choose a stable ID-hash sample while counting every S1 source row."""
    if divisor < 1:
        raise ValueError("sample divisor must be positive")
    parts = []
    total = 0
    for chunk in iter_source(path, chunk_size):
        total += len(chunk)
        hashes = pd.util.hash_pandas_object(chunk["entity_id"], index=False).to_numpy(dtype="uint64")
        selected = chunk.loc[hashes % divisor == 0]
        if len(selected):
            parts.append(selected)
    if not parts:
        raise ValueError("S1 sample is empty; reduce sample_divisor")
    sample = pd.concat(parts, ignore_index=True)
    if sample["entity_id"].duplicated().any():
        raise ValueError("Duplicate S1 entity_id in the selected sample")
    return sample, total


def load_ground_truth(path: Path, wanted: set[str], chunk_size: int) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for chunk in pd.read_csv(path, sep="\t", chunksize=chunk_size,
                             dtype="string", keep_default_na=False):
        if list(chunk.columns) != TRUTH_COLUMNS:
            raise ValueError(f"Unexpected ground-truth columns in {path}: {list(chunk.columns)}")
        selected = chunk.loc[chunk["source1_entity_id"].isin(wanted)]
        for entity_id, raw in zip(selected["source1_entity_id"], selected["matched_entity_ids"]):
            if entity_id in result:
                raise ValueError(f"Duplicate ground-truth row: {entity_id}")
            links = [part.strip() for part in str(raw).split(",") if part.strip()]
            if len(links) != len(set(links)):
                raise ValueError(f"Duplicate linked ID in ground truth: {entity_id}")
            result[str(entity_id)] = set(links)
    if set(result) != wanted:
        raise ValueError(f"Ground truth missing {len(wanted - set(result))} selected S1 IDs")
    return result
