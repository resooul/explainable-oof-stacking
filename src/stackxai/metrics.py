"""Point-forecast error metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd

METRIC_ORDER = ["RMSE", "MAE", "MedAE", "MAPE", "SMAPE", "R2", "NRMSE"]


def rmse(y, p):
    """Root mean squared error."""
    return float(np.sqrt(np.mean((y - p) ** 2)))


def mae(y, p):
    """Mean absolute error."""
    return float(np.mean(np.abs(y - p)))


def mape(y, p):
    """Mean absolute percentage error in percent; rows with a zero target are skipped."""
    m = y != 0
    return float(np.mean(np.abs((y[m] - p[m]) / y[m])) * 100)


def smape(y, p):
    """Symmetric mean absolute percentage error in percent; a row whose actual and predicted values are both zero
    counts as zero.
    """
    d = (np.abs(y) + np.abs(p)) / 2
    r = np.divide(np.abs(y - p), d, out=np.zeros_like(d, dtype=float), where=d != 0)
    return float(np.mean(r) * 100)


def r2(y, p):
    """Coefficient of determination."""
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - np.mean(y)) ** 2))


# Metrics accepted by tuning.metric, stacking.selection_metric and evaluation.primary_metric.
FUNCS = {"rmse": rmse, "mae": mae, "mape": mape, "smape": smape, "r2": r2}
LOWER_IS_BETTER = {"rmse": True, "mae": True, "mape": True, "smape": True, "r2": False}


def score(metric: str, y, p) -> float:
    """Loss-oriented score (lower is better) used for tuning and model selection."""
    v = FUNCS[metric](np.asarray(y, float), np.asarray(p, float))
    return v if LOWER_IS_BETTER[metric] else -v


def compute(y, p) -> dict:
    """Every metric of ``p`` against ``y`` on the rows with a finite prediction. NRMSE is the RMSE divided by the
    mean target of those rows.
    """
    y, p = np.asarray(y, float), np.asarray(p, float)
    m = np.isfinite(p)
    y, p = y[m], p[m]
    e = rmse(y, p)
    return {"n": int(m.sum()), "RMSE": e, "MAE": mae(y, p), "MedAE": float(np.median(np.abs(y - p))),
            "MAPE": mape(y, p), "SMAPE": smape(y, p), "R2": r2(y, p), "NRMSE": e / float(np.mean(y))}


def table(y, preds: dict[str, np.ndarray], **extra) -> pd.DataFrame:
    """One row of metrics per model in ``preds``; ``extra`` adds columns with constant values."""
    rows = [{"model": k, **extra, **compute(y, v)} for k, v in preds.items()]
    return pd.DataFrame(rows)


def by_group(y, preds: dict[str, np.ndarray], groups: np.ndarray, label: str) -> pd.DataFrame:
    """Metric table for every value of ``groups``, recorded in the column ``label``."""
    frames = []
    for g in pd.unique(groups):
        m = groups == g
        if m.sum() == 0:
            continue
        frames.append(table(y[m], {k: v[m] for k, v in preds.items()}, **{label: g}))
    return pd.concat(frames, ignore_index=True)
