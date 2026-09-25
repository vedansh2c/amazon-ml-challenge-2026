"""First complete candidate, matcher, validation and submission workflow."""

from __future__ import annotations

import csv
import json
import logging
import zlib
from collections import Counter
from itertools import islice
from multiprocessing import get_context
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from lightgbm.basic import LightGBMError
from rapidfuzz import fuzz

from src.data.index import build_index, iter_records, open_index, source_path
from src.features.text import FEATURE_NAMES, features, keys, normalize

LOG = logging.getLogger(__name__)
_WORKER = {}


def sample_anchor(entity_id: str, divisor: int) -> bool:
    return zlib.crc32(entity_id.encode("utf-8")) % divisor == 0


def load_sampled_anchors(dataset: Path, divisor: int) -> list[tuple[str, str, str, str]]:
    anchors = [row for row in iter_records(source_path(dataset, "train", 1))
               if sample_anchor(row[0], divisor)]
    if not anchors:
        raise ValueError("No training S1 records selected; decrease --sample-divisor")
    LOG.info("Selected %s training S1 records", f"{len(anchors):,}")
    return anchors


def load_truth(dataset: Path, wanted_ids: set[str]) -> dict[str, set[str]]:
    path = dataset / "train" / "train_ground_truth.tsv"
    result = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if set(reader.fieldnames or []) != {"source1_entity_id", "matched_entity_ids"}:
            raise ValueError(f"Unexpected ground-truth columns: {reader.fieldnames}")
        for row in reader:
            entity_id = row["source1_entity_id"]
            if entity_id in wanted_ids:
                if entity_id in result:
                    raise ValueError(f"Duplicate ground-truth row for {entity_id}")
                result[entity_id] = {value.strip() for value in row["matched_entity_ids"].split(",")
                                     if value.strip()}
    if result.keys() != wanted_ids:
        raise ValueError(f"Ground truth missing {len(wanted_ids - result.keys())} selected S1 IDs")
    return result


def generate_candidates(conn, anchor: tuple[str, str, str, str], *, per_key: int = 30,
                        final_limit: int = 50) -> list[tuple[str, str, str, str]]:
    """Return exactly the pairs that the classifier will later score."""
    name_key, token_key, address_key = keys(anchor[1], anchor[2])
    found: dict[str, tuple[str, str, str, str]] = {}
    for field, key in (("name_key", name_key), ("token_key", token_key),
                       ("address_key", address_key)):
        if not key:
            continue
        # Query sources separately so a common block cannot hide all S3 records.
        for source in ("S2", "S3"):
            rows = conn.execute(f"SELECT entity_id, name, address, country FROM records "
                                f"WHERE country=? AND {field}=? AND source=? LIMIT ?",
                                (anchor[3], key, source, per_key)).fetchall()
            for row in rows:
                found[row[0]] = row
    if len(found) <= final_limit:
        return list(found.values())
    name = normalize(anchor[1], name=True)
    address = normalize(anchor[2])
    scored = []
    for row in found.values():
        name_score = fuzz.ratio(name, normalize(row[1], name=True))
        address_score = fuzz.ratio(address, normalize(row[2]))
        scored.append((max(name_score, address_score), name_score + address_score, row[0], row))
    scored.sort(reverse=True)
    source_quota = final_limit // 2
    selected = [item[3] for source in ("S2-", "S3-")
                for item in [row for row in scored if row[2].startswith(source)][:source_quota]]
    selected_ids = {row[0] for row in selected}
    for item in scored:
        if len(selected) >= final_limit:
            break
        if item[2] not in selected_ids:
            selected.append(item[3])
            selected_ids.add(item[2])
    return selected


def candidate_features(anchor, candidates):
    if not candidates:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
    return np.asarray([features(anchor, candidate) for candidate in candidates], dtype=np.float32)


def macro_f05(truth: dict[str, set[str]], predictions: dict[str, set[str]]) -> float:
    scores = []
    for entity_id, actual in truth.items():
        predicted = predictions.get(entity_id, set())
        if not actual and not predicted:
            scores.append(1.0)
        elif not actual or not predicted:
            scores.append(0.0)
        else:
            common = len(actual & predicted)
            precision = common / len(predicted)
            recall = common / len(actual)
            scores.append(1.25 * precision * recall / (0.25 * precision + recall)
                          if common else 0.0)
    return float(np.mean(scores))


def _fit_model(X: np.ndarray, y: np.ndarray, requested_device: str):
    if requested_device not in {"auto", "cpu", "gpu"}:
        raise ValueError("device must be auto, cpu, or gpu")
    base_params = dict(n_estimators=160, learning_rate=0.05, num_leaves=15,
                       max_depth=6, min_child_samples=40, random_state=2026,
                       verbosity=-1, n_jobs=4)
    devices = ["gpu", "cpu"] if requested_device == "auto" else [requested_device]
    last_error = None
    for device in devices:
        try:
            model = LGBMClassifier(**base_params, device_type=device)
            model.fit(pd.DataFrame(X, columns=FEATURE_NAMES), y)
            LOG.info("Fitted LightGBM using %s", device)
            return model, device
        except (LightGBMError, RuntimeError, ValueError) as error:
            last_error = error
            if device == "gpu" and requested_device == "auto":
                LOG.warning("GPU LightGBM unavailable; falling back to CPU: %s", error)
                continue
            raise
    raise RuntimeError(f"Could not fit LightGBM: {last_error}")


def train(dataset: Path, output: Path, *, sample_divisor: int = 100,
          per_key: int = 30, final_limit: int = 50, device: str = "auto") -> dict:
    output.mkdir(parents=True, exist_ok=True)
    index_path = output / "train_candidates.sqlite"
    build_index(dataset, "train", index_path)
    anchors = load_sampled_anchors(dataset, sample_divisor)
    truth = load_truth(dataset, {row[0] for row in anchors})
    training_x, training_y = [], []
    calibration = []
    validation = []
    recall_counts = Counter()
    conn = open_index(index_path)
    try:
        for number, anchor in enumerate(anchors, 1):
            candidates = generate_candidates(conn, anchor, per_key=per_key, final_limit=final_limit)
            vector = candidate_features(anchor, candidates)
            actual = truth[anchor[0]]
            retrieved = {row[0] for row in candidates}
            fold = zlib.crc32((anchor[0] + "fold").encode()) % 10
            partition = "calibration" if fold == 0 else "validation" if fold == 1 else "train"
            recall_counts[f"{partition}_links"] += len(actual)
            recall_counts[f"{partition}_recalled"] += len(actual & retrieved)
            recall_counts[f"{partition}_s1"] += 1
            recall_counts[f"{partition}_candidates"] += len(candidates)
            if partition == "calibration":
                calibration.append((anchor[0], actual, candidates, vector))
            elif partition == "validation":
                validation.append((anchor[0], actual, candidates, vector))
            elif candidates:
                labels = np.array([int(row[0] in actual) for row in candidates], dtype=np.int8)
                # Include every retrieved positive and difficult negatives from both sources.
                negatives_by_source = []
                for prefix in ("S2-", "S3-"):
                    negatives_by_source.extend(i for i, row in enumerate(candidates)
                                               if row[0].startswith(prefix) and labels[i] == 0)
                    if prefix == "S2-":
                        s2_count = len(negatives_by_source)
                negatives = np.asarray(negatives_by_source[:6] +
                                       negatives_by_source[s2_count:s2_count + 6], dtype=int)
                chosen = np.concatenate([np.flatnonzero(labels == 1), negatives])
                training_x.append(vector[chosen])
                training_y.append(labels[chosen])
            if number % 2_000 == 0:
                LOG.info("Processed %s sampled S1 anchors", f"{number:,}")
    finally:
        conn.close()
    if not training_x:
        raise ValueError("No training candidates; adjust blocking keys")
    X = np.vstack(training_x)
    y = np.concatenate(training_y)
    if len(np.unique(y)) != 2:
        raise ValueError("Training pairs need both classes; adjust blocking/sample size")
    model, selected_device = _fit_model(X, y, device)
    LOG.info("Fitted LightGBM on %s pairs (%s positives)", f"{len(y):,}", f"{int(y.sum()):,}")
    def score_anchors(group):
        scored = []
        for entity_id, actual, candidates, vector in group:
            probabilities = (model.predict_proba(pd.DataFrame(vector, columns=FEATURE_NAMES))[:, 1]
                             if len(candidates) else np.array([]))
            scored.append((entity_id, actual, candidates, probabilities))
        return scored

    scored_calibration = score_anchors(calibration)
    scored_validation = score_anchors(validation)
    calibration_truth = {entity_id: actual for entity_id, actual, _, _ in scored_calibration}
    valid_truth = {entity_id: actual for entity_id, actual, _, _ in scored_validation}
    thresholds = np.arange(0.05, 0.951, 0.05)
    threshold_scores = {}
    for threshold in thresholds:
        predictions = {entity_id: {row[0] for row, probability in zip(candidates, probabilities)
                                   if probability >= threshold}
                       for entity_id, _, candidates, probabilities in scored_calibration}
        threshold_scores[round(float(threshold), 2)] = macro_f05(calibration_truth, predictions)
    best_threshold = max(threshold_scores, key=lambda value: (threshold_scores[value], value))
    best_predictions = {entity_id: {row[0] for row, probability in zip(candidates, probabilities)
                                    if probability >= best_threshold}
                        for entity_id, _, candidates, probabilities in scored_validation}
    true_positive_links = sum(len(valid_truth[entity_id] & predicted)
                              for entity_id, predicted in best_predictions.items())
    predicted_links = sum(map(len, best_predictions.values()))
    singleton_ids = {entity_id for entity_id, actual in valid_truth.items() if not actual}
    source_recall = {}
    for prefix in ("S2-", "S3-"):
        denominator = sum(sum(candidate.startswith(prefix) for candidate in actual)
                          for actual in valid_truth.values())
        numerator = sum(sum(candidate.startswith(prefix) for candidate in
                            actual & {row[0] for row in candidates})
                        for _, actual, candidates, _ in scored_validation)
        source_recall[prefix[:2]] = numerator / denominator if denominator else None
    metrics = {
        "sample_divisor": sample_divisor, "per_key": per_key, "final_limit": final_limit,
        "requested_device": device, "selected_device": selected_device,
        "train_s1": recall_counts["train_s1"],
        "calibration_s1": recall_counts["calibration_s1"],
        "validation_s1": recall_counts["validation_s1"],
        "training_pairs": int(len(y)), "training_positives": int(y.sum()),
        "train_candidate_recall": recall_counts["train_recalled"] / max(1, recall_counts["train_links"]),
        "validation_candidate_recall": recall_counts["validation_recalled"] / max(1, recall_counts["validation_links"]),
        "validation_true_links": recall_counts["validation_links"],
        "validation_average_candidates": recall_counts["validation_candidates"] / max(1, recall_counts["validation_s1"]),
        "calibration_macro_f05": threshold_scores[best_threshold],
        "validation_macro_f05": macro_f05(valid_truth, best_predictions),
        "threshold": best_threshold, "calibration_threshold_scores": threshold_scores,
        "validation_pair_precision": true_positive_links / predicted_links if predicted_links else 0.0,
        "validation_pair_recall": true_positive_links / max(1, recall_counts["validation_links"]),
        "validation_singletons": len(singleton_ids),
        "validation_singleton_accuracy": (sum(not best_predictions[entity_id] for entity_id in singleton_ids)
                                          / len(singleton_ids) if singleton_ids else None),
        "validation_source_candidate_recall": source_recall,
    }
    joblib.dump(model, output / "baseline_model.joblib")
    (output / "baseline_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    LOG.info("Validation: candidate recall %.4f, macro F0.5 %.4f at threshold %.2f",
             metrics["validation_candidate_recall"], metrics["validation_macro_f05"], best_threshold)
    return metrics


def _init_prediction_worker(index_path: Path, model_path: Path, threshold: float,
                            per_key: int, final_limit: int) -> None:
    model = joblib.load(model_path)
    model.set_params(n_jobs=1)
    _WORKER.update(conn=open_index(index_path), model=model, threshold=threshold,
                   per_key=per_key, final_limit=final_limit)


def _predict_batch(anchors):
    conn = _WORKER["conn"]
    model = _WORKER["model"]
    batch = []
    matrices = []
    for anchor in anchors:
        candidates = generate_candidates(conn, anchor, per_key=_WORKER["per_key"],
                                         final_limit=_WORKER["final_limit"])
        batch.append((anchor[0], candidates))
        if candidates:
            matrices.append(candidate_features(anchor, candidates))
    scores = (model.predict_proba(pd.DataFrame(np.vstack(matrices), columns=FEATURE_NAMES))[:, 1]
              if matrices else np.array([]))
    offset = 0
    result = []
    for entity_id, candidates in batch:
        count = len(candidates)
        probabilities = scores[offset:offset + count]
        offset += count
        candidate_ids = [row[0] for row in candidates]
        matched_ids = [row[0] for row, probability in zip(candidates, probabilities)
                       if probability >= _WORKER["threshold"]]
        result.append((entity_id, candidate_ids, matched_ids))
    return result


def _anchor_batches(dataset: Path, size: int = 500):
    records = iter(iter_records(source_path(dataset, "test", 1)))
    while batch := list(islice(records, size)):
        yield batch


def predict(dataset: Path, output: Path, *, per_key: int = 30, final_limit: int = 50,
            workers: int = 4) -> None:
    metrics = json.loads((output / "baseline_metrics.json").read_text())
    if (metrics["per_key"], metrics["final_limit"]) != (per_key, final_limit):
        raise ValueError("Prediction candidate settings must match training settings")
    index_path = output / "test_candidates.sqlite"
    build_index(dataset, "test", index_path)
    threshold = metrics["threshold"]
    output_dir = output / "submission"
    output_dir.mkdir(parents=True, exist_ok=True)
    matching_path = output_dir / "matching_results.tsv"
    candidate_path = output_dir / "candidate_pairs.tsv"
    if workers < 1:
        raise ValueError("workers must be at least 1")
    from contextlib import nullcontext
    pool_context = (get_context("spawn").Pool(workers, initializer=_init_prediction_worker,
                    initargs=(index_path, output / "baseline_model.joblib", threshold,
                              per_key, final_limit)) if workers > 1 else nullcontext())
    if workers == 1:
        _init_prediction_worker(index_path, output / "baseline_model.joblib", threshold,
                                per_key, final_limit)
    try:
      with pool_context as pool:
        with matching_path.open("w", newline="", encoding="utf-8") as matching_file, \
             candidate_path.open("w", newline="", encoding="utf-8") as candidate_file:
            matching_writer = csv.writer(matching_file, delimiter="\t", lineterminator="\n")
            candidate_writer = csv.writer(candidate_file, delimiter="\t", lineterminator="\n")
            matching_writer.writerow(["source1_entity_id", "matched_entity_ids"])
            candidate_writer.writerow(["source1_entity_id", "candidate_entity_ids"])
            batches = _anchor_batches(dataset)
            results = pool.imap(_predict_batch, batches, chunksize=1) if pool else map(_predict_batch, batches)
            number = 0
            for result in results:
                for entity_id, candidate_ids, matched_ids in result:
                    candidate_writer.writerow([entity_id, ",".join(candidate_ids)])
                    matching_writer.writerow([entity_id, ",".join(matched_ids)])
                number += len(result)
                if number % 100_000 == 0:
                    LOG.info("Wrote %s test S1 predictions", f"{number:,}")
    finally:
        if workers == 1:
            _WORKER["conn"].close()
    LOG.info("Wrote %s and %s", matching_path, candidate_path)
