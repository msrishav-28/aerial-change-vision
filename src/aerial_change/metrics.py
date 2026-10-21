from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score


def binary_report(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    out = {"n": float(len(y_true)), "prevalence": float(y_true.mean()) if len(y_true) else float("nan")}
    if len(y_true) and y_true.min() != y_true.max():
        out["auprc"] = float(average_precision_score(y_true, y_score))
    else:
        out["auprc"] = float("nan")
    return out


def select_threshold_at_fpr(
    y_true: np.ndarray,
    y_score: np.ndarray,
    max_fpr: float = 0.05,
) -> float | None:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    n_neg = int((1 - y_true).sum())
    if n_neg == 0:
        return None
    order = np.argsort(-y_score)
    fp = np.cumsum(1 - y_true[order])
    fpr = fp / n_neg
    within_budget = np.where(fpr <= max_fpr)[0]
    if len(within_budget) == 0:
        return None
    return float(y_score[order][within_budget[-1]])


def precision_at_threshold(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    flagged = y_score >= threshold
    n_flagged = int(flagged.sum())
    if n_flagged == 0:
        return float("nan")
    tp = int(y_true[flagged].sum())
    return float(tp / n_flagged)


def precision_at_fpr(
    y_true: np.ndarray,
    y_score: np.ndarray,
    max_fpr: float = 0.05,
) -> float:
    """Operating point chosen on the SAME set; for test reporting select on
    validation with select_threshold_at_fpr, then use precision_at_threshold."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    if len(y_true) == 0 or y_true.min() == y_true.max():
        return float("nan")
    threshold = select_threshold_at_fpr(y_true, y_score, max_fpr)
    if threshold is None:
        return float("nan")
    return precision_at_threshold(y_true, y_score, threshold)


def mask_iou(pred: np.ndarray, target: np.ndarray, thresh: float = 0.5) -> float:
    p = pred >= thresh
    t = target >= 0.5
    inter = np.logical_and(p, t).sum()
    union = np.logical_or(p, t).sum()
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return float(inter / union)
