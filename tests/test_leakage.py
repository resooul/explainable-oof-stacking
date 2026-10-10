"""Leakage safeguards: vocabularies fitted on training rows only and out-of-fold predictions never in-sample."""
import numpy as np
import pandas as pd

from stackxai.config import DEFAULTS, deep_merge
from stackxai.data import Dataset
from stackxai.models import TabularEncoder
from stackxai.splits import holdout, temporal_folds
from stackxai.stacking import run_base


def test_encoder_vocabulary_is_fitted_on_training_rows_only():
    train = pd.DataFrame({"c": ["a", "b", "a"], "x": [1.0, 2.0, 3.0]})
    test = pd.DataFrame({"c": ["a", "z"], "x": [1.0, 1.0]})
    enc = TabularEncoder(["c", "x"], ["c"]).fit(train)
    codes = enc.to_numeric(test)[:, 0]
    assert codes[0] == 0 and np.isnan(codes[1])


def test_out_of_fold_predictions_are_not_in_sample():
    rng = np.random.default_rng(0)
    n = 3000
    frame = pd.DataFrame({"t": pd.date_range("2021-01-01", periods=n, freq="min"),
                          "x": rng.normal(size=n), "y": rng.normal(size=n)})
    ds = Dataset(frame=frame, features=["x"], categorical=[], target="y", time_col="t")
    cfg = deep_merge(DEFAULTS, {"run": {"n_jobs": 1}})
    train, test = holdout(n, ds.unit_starts(), 0.3)
    folds = temporal_folds(train, ds.unit_starts(), 5, "blocked")
    res = run_base("DT", ds, train, test, folds, cfg, 0, "none", 0, "blocked", False,
                   fixed=([{"min_samples_leaf": 1}] * 5, {"min_samples_leaf": 1}))
    oof_rmse = np.sqrt(np.mean((res.oof - ds.y[train]) ** 2))
    assert oof_rmse > 0.9 * ds.y[train].std()
