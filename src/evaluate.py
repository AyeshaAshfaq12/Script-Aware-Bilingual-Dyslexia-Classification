"""Phase 4/5: metrics, pooled and per script (guide sections 9, 10).

Every metric here is computed from saved per-image predictions, so any
number in the paper can be traced back to a run artifact on disk.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score, precision_score, recall_score,
                             roc_auc_score)

SCRIPTS = ("urdu", "english", "digit")


def _safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """AUC is undefined when a subset holds a single class."""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def _block(y_true: np.ndarray, y_prob: np.ndarray,
           threshold: float) -> dict[str, float]:
    if len(y_true) == 0:
        return {k: float("nan") for k in
                ("acc", "bal_acc", "prec", "rec", "f1", "auc")}
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "bal_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "prec": float(precision_score(y_true, y_pred, zero_division=0)),
        "rec": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": _safe_auc(y_true, y_prob),
    }


def compute_metrics(y_true, y_prob, scripts, threshold: float = 0.5) -> dict:
    """Pooled metrics plus a per-script breakdown.

    y_true  (n,) 0/1
    y_prob  (n,) predicted probability of class 1 (dyslexic)
    scripts (n,) one of SCRIPTS per image
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    scripts = np.asarray(scripts)
    assert y_true.shape == y_prob.shape == scripts.shape, "length mismatch"

    out: dict[str, float] = {"n": int(len(y_true))}
    out.update(_block(y_true, y_prob, threshold))
    for s in SCRIPTS:
        m = scripts == s
        blk = _block(y_true[m], y_prob[m], threshold)
        out[f"n_{s}"] = int(m.sum())
        for k, v in blk.items():
            out[f"{k}_{s}"] = v
    return out


def script_classifier_metrics(y_true_ids, y_pred_ids) -> dict:
    y_true_ids = np.asarray(y_true_ids).ravel()
    y_pred_ids = np.asarray(y_pred_ids).ravel()
    out = {"script_acc": float(accuracy_score(y_true_ids, y_pred_ids)),
           "script_f1_macro": float(f1_score(y_true_ids, y_pred_ids,
                                             average="macro",
                                             zero_division=0))}
    for i, s in enumerate(SCRIPTS):
        m = y_true_ids == i
        out[f"script_acc_{s}"] = (float((y_pred_ids[m] == i).mean())
                                  if m.any() else float("nan"))
    return out
