"""Target transformation for training and retransformation for reporting.

Models are trained, tuned and selected on the *model scale*; every reported metric, table and exported
prediction is on the *report scale* (``dataset.target_unit``).

``dataset.target_transform``
    ``none``  model scale = report scale.
    ``log``   model scale = log of the target in its original unit (before ``target_scale``), e.g. log(minutes)
              while reporting in hours. For tree models the unit inside the logarithm is immaterial (it only
              shifts the target by a constant).

Retransformation bias. A model fitted by squared error on log(y) estimates E[log y | x]; exp() of that is a
conditional median, which under-predicts the conditional mean of a right-skewed target. Duan's smearing
estimator corrects this with the factor mean(exp(residual)), computed here from out-of-sample residuals on the
training partition only (out-of-fold predictions for base learners; predictions on the combiner-selection
windows for combiners), never from the test partition. ``dataset.target_smearing``:
    ``stratum``  one factor per stratum (``dataset.strata_col``, e.g. pickup / delivery), since residual
                 dispersion differs between strata; falls back to the overall factor for unseen strata;
    ``global``   one overall factor;
    ``none``     no correction (the retransformed prediction is a conditional median).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Accepted values of dataset.target_transform and dataset.target_smearing.
TRANSFORMS = ("none", "log")
SMEARING = ("stratum", "global", "none")
# Key of the smearing factor estimated on all rows; it is used for strata without a factor of their own.
ALL = "_all_"


@dataclass
class TargetScale:
    """Mapping between the report scale and the model scale of the target (see the module docstring)."""
    kind: str = "none"
    scale: float = 1.0          # report unit = original unit * scale

    def forward(self, y_report: np.ndarray) -> np.ndarray:
        """Report scale -> model scale."""
        y_report = np.asarray(y_report, dtype=float)
        if self.kind == "none":
            return y_report
        return np.log(y_report / self.scale)

    def inverse(self, p_model: np.ndarray) -> np.ndarray:
        """Model scale -> report scale, without bias correction."""
        p_model = np.asarray(p_model, dtype=float)
        if self.kind == "none":
            return p_model
        return np.exp(p_model) * self.scale


def smearing_factors(y_model: np.ndarray, p_model: np.ndarray, strata=None, mode: str = "stratum") -> dict:
    """Duan's smearing factors mean(exp(y - p)) on rows where ``p`` is finite."""
    y_model, p_model = np.asarray(y_model, float), np.asarray(p_model, float)
    ok = np.isfinite(p_model) & np.isfinite(y_model)
    if mode == "none" or not ok.any():
        return {ALL: 1.0}
    r = np.exp(y_model[ok] - p_model[ok])
    out = {ALL: float(r.mean())}
    if mode == "stratum" and strata is not None:
        s = np.asarray(strata, dtype=object)[ok]
        for v in pd.unique(s):
            out[v] = float(r[s == v].mean())
    return out


def to_report(p_model: np.ndarray, ts: TargetScale, factors: dict | None = None, strata=None,
              clip_min: float | None = None) -> np.ndarray:
    """Model-scale predictions -> report-scale predictions (smearing-corrected, clipped)."""
    p = ts.inverse(p_model)
    if ts.kind != "none" and factors:
        if strata is not None and len(factors) > 1:
            f = pd.Series(np.asarray(strata, dtype=object)).map(factors).fillna(factors[ALL]).to_numpy(float)
        else:
            f = factors[ALL]
        p = p * f
    return np.maximum(p, clip_min) if clip_min is not None else p
