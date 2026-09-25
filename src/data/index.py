"""Disk-backed candidate index over complete S2 and S3 source files."""

from __future__ import annotations

import csv
import json
import logging
import sqlite3
from pathlib import Path

from src.features.text import keys

LOG = logging.getLogger(__name__)
SCHEMA_VERSION = 2


def source_path(dataset: Path, split: str, source: int) -> Path:
    return dataset / split / f"{split}_source{source}.tsv"


def iter_records(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        expected = {"entity_id", "business_name", "business_address", "country"}
        if set(reader.fieldnames or []) != expected:
            raise ValueError(f"Unexpected TSV columns in {path}: {reader.fieldnames}")
        for row in reader:
            yield (row["entity_id"], row["business_name"], row["business_address"], row["country"])


def index_signature(dataset: Path, split: str) -> dict:
    result = {"schema": SCHEMA_VERSION, "split": split}
    for source in (2, 3):
        path = source_path(dataset, split, source)
        stat = path.stat()
        result[f"S{source}"] = [str(path.resolve()), stat.st_size, stat.st_mtime_ns]
    return result


def open_index(db_path: Path, *, readonly: bool = True) -> sqlite3.Connection:
    if readonly:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    else:
        conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA cache_size=-65536")
    return conn


def build_index(dataset: Path, split: str, db_path: Path) -> None:
    """Reuse a complete matching index, or construct it without changing raw TSVs."""
    signature = index_signature(dataset, split)
    metadata_path = db_path.with_suffix(".json")
    if db_path.exists() and metadata_path.exists():
        if json.loads(metadata_path.read_text()) == signature:
            LOG.info("Reusing %s index: %s", split, db_path)
            return
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    metadata_path.unlink(missing_ok=True)
    conn = open_index(db_path, readonly=False)
    conn.execute("PRAGMA journal_mode=OFF")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("CREATE TABLE records (entity_id TEXT NOT NULL, source TEXT NOT NULL, name TEXT NOT NULL, "
                 "address TEXT NOT NULL, country TEXT NOT NULL, "
                 "name_key TEXT, token_key TEXT, address_key TEXT)")
    total = 0
    try:
        for source in (2, 3):
            batch = []
            for entity_id, name, address, country in iter_records(source_path(dataset, split, source)):
                key1, key2, key3 = keys(name, address)
                batch.append((entity_id, f"S{source}", name, address, country, key1, key2, key3))
                if len(batch) >= 25_000:
                    conn.executemany("INSERT INTO records VALUES (?,?,?,?,?,?,?,?)", batch)
                    conn.commit()
                    total += len(batch)
                    if total % 500_000 == 0:
                        LOG.info("%s index: %s records", split, f"{total:,}")
                    batch.clear()
            if batch:
                conn.executemany("INSERT INTO records VALUES (?,?,?,?,?,?,?,?)", batch)
                conn.commit()
                total += len(batch)
        for field in ("name_key", "token_key", "address_key"):
            LOG.info("Creating %s index for %s", field, split)
            conn.execute(f"CREATE INDEX idx_{field} ON records(country, {field}, source)")
        conn.commit()
        metadata_path.write_text(json.dumps(signature, indent=2) + "\n")
        LOG.info("Built %s index: %s records", split, f"{total:,}")
    finally:
        conn.close()
