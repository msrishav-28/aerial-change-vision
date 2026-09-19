from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve


def binary_report(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    out = {"n": float(len(y_true)), "prevalence": float(y_true.mean()) if len(y_true) else float("nan")}
    if len(y_true) and y_true.min() != y_true.max():
        out["auprc"] = float(average_precision_score(y_true, y_score))
    else:
        out["auprc"] = float("nan")
    return out


def precision_at_fpr(
    y_true: np.ndarray,
    y_score: np.ndarray,
    max_fpr: float = 0.05,
) -> float:
    """Precision when false-positive rate is at most max_fpr (analyst budget)."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    if y_true.min() == y_true.max():
        return float("nan")
    prec, rec, thr = precision_recall_curve(y_true, y_score)
    # approximate FPR from scores descending
    order = np.argsort(-y_score)
    yt = y_true[order]
    fp = np.cumsum(1 - yt)
    tn_fp = max((1 - y_true).sum(), 1)
    fpr = fp / tn_fp
    ok = np.where(fpr <= max_fpr)[0]
    if len(ok) == 0:
        return float("nan")
    tp = np.cumsum(yt)
    pp = np.maximum(tp + fp, 1)
    precision = tp / pp
    return float(precision[ok[-1]])


def mask_iou(pred: np.ndarray, target: np.ndarray, thresh: float = 0.5) -> float:
    p = pred >= thresh
    t = target >= 0.5
    inter = np.logical_and(p, t).sum()
    union = np.logical_or(p, t).sum()
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return float(inter / union)
