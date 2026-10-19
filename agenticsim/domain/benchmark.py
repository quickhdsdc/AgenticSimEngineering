"""Dependency-free grouped splits and deterministic anomaly scoring."""
import hashlib
import math
import statistics

from agenticsim.domain import require


def grouped_split(records, seed):
    groups = sorted({r["workpiece_id"] for r in records},
                    key=lambda key: hashlib.sha256(f"{seed}:{key}".encode()).hexdigest())
    require(len(groups) >= 5, "not enough workpieces for three splits")
    train_end, validation_end = int(.6 * len(groups)), int(.8 * len(groups))
    assignment = {g: "train" if i < train_end else "validation" if i < validation_end else "test"
                  for i, g in enumerate(groups)}
    return {part: [r for r in records if assignment[r["workpiece_id"]] == part]
            for part in ("train", "validation", "test")}


def fit_normal(records, labels, names, scale_floors):
    normal = [r for r in records if labels[r["operation_id"]] == 0]
    require(len(normal) > 1, "normal training examples required")
    center, scale = {}, {}
    for name in names:
        values = [r["features"][name] for r in normal]
        center[name] = statistics.median(values)
        scale[name] = max(1.4826 * statistics.median(abs(v - center[name]) for v in values), scale_floors[name])
    return {"center": center, "scale": scale}


def anomaly_score(record, model):
    return max(abs(record["features"][name] - center) / model["scale"][name]
               for name, center in model["center"].items())


def classification_metrics(labels, scores, threshold):
    require(len(labels) == len(scores) and bool(labels), "invalid metric arrays")
    require(all(y in (0, 1) for y in labels) and all(math.isfinite(s) for s in scores), "invalid labels or scores")
    tp = sum(y == 1 and s >= threshold for y, s in zip(labels, scores))
    fp = sum(y == 0 and s >= threshold for y, s in zip(labels, scores))
    fn, tn = sum(labels) - tp, len(labels) - sum(labels) - fp
    precision, recall = tp / (tp + fp) if tp + fp else 0., tp / (tp + fn) if tp + fn else 0.
    # Group tied scores before updating precision: average precision, not
    # trapezoidal PR-AUC. This avoids optimistic tie ordering.
    grouped = {}
    for y, s in zip(labels, scores):
        n, p = grouped.get(s, (0, 0))
        grouped[s] = n + 1, p + y
    cumulative, positives, ap = 0, 0, 0.
    for score in sorted(grouped, reverse=True):
        n, p = grouped[score]
        cumulative += n
        positives += p
        if sum(labels):
            ap += (p / sum(labels)) * positives / cumulative
    return {"operations": len(labels), "positives": sum(labels), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.,
            "false_positive_rate": fp / (fp + tn) if fp + tn else 0., "average_precision": ap}


def choose_threshold(labels, scores, candidates):
    require(set(labels) == {0, 1}, "validation requires both classes")
    # Candidate grid is fixed before scoring; ties prefer the higher threshold.
    return max(candidates, key=lambda t: (classification_metrics(labels, scores, t)["f1"], t))
