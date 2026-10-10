"""Target transformation and smearing retransformation."""
import joblib
import numpy as np
import pandas as pd
import pytest
import yaml

from stackxai.config import load_config
from stackxai.pipeline import run
from stackxai.target import ALL, TargetScale, smearing_factors, to_report


def test_log_round_trip_and_unit():
    ts = TargetScale("log", 1 / 60)                     # model scale log(minutes), report scale hours
    hours = np.array([1 / 60, 0.5, 2.0, 30.0])
    assert np.allclose(ts.forward(hours), np.log(hours * 60))
    assert np.allclose(ts.inverse(ts.forward(hours)), hours)
    assert (ts.forward(hours) >= 0).all()               # log(minutes) is non-negative for durations >= 1 minute


def test_none_is_the_identity():
    ts = TargetScale()
    x = np.array([-1.0, 0.0, 2.5])
    assert np.array_equal(ts.forward(x), x) and np.array_equal(ts.inverse(x), x)
    assert np.array_equal(to_report(x, ts, {ALL: 3.0}, None, 0.0), np.maximum(x, 0.0))


def test_smearing_recovers_the_lognormal_mean():
    rng = np.random.default_rng(0)
    n = 200_000
    s = np.where(rng.random(n) < 0.5, "pickup", "delivery")
    sigma = np.where(s == "pickup", 0.2, 0.8)
    mu = rng.normal(3, 0.5, n)
    y = mu + rng.normal(0, sigma)                      # log scale
    f = smearing_factors(y, mu, s, "stratum")
    assert f["pickup"] == pytest.approx(np.exp(0.2 ** 2 / 2), rel=0.01)
    assert f["delivery"] == pytest.approx(np.exp(0.8 ** 2 / 2), rel=0.01)
    ts = TargetScale("log", 1.0)
    corrected = to_report(mu, ts, f, s)
    for v in ("pickup", "delivery"):                   # unbiased on the original scale, per stratum
        m = s == v
        assert corrected[m].mean() == pytest.approx(np.exp(y[m]).mean(), rel=0.01)
    assert to_report(mu, ts, smearing_factors(y, mu, s, "none"), s).mean() < np.exp(y).mean() * 0.9


def test_unseen_stratum_falls_back_to_overall_factor():
    f = {ALL: 2.0, "a": 1.5}
    out = to_report(np.zeros(3), TargetScale("log", 1.0), f, np.array(["a", "b", None], dtype=object))
    assert np.allclose(out, [1.5, 2.0, 2.0])


def _table(tmp_path, n_orders=1500, seed=0):
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp("2021-01-01") + pd.to_timedelta(np.sort(rng.uniform(0, 300 * 86400, n_orders)), unit="s")
    rows = []
    for i in range(n_orders):
        carrier = int(rng.integers(0, 6))
        dist = float(rng.gamma(2, 6))
        pick = float(rng.integers(1, 6)) if rng.random() < 0.4 else float(np.exp(rng.normal(3, 0.6)))
        deliv = pick + float(np.exp(rng.normal(4 + 0.1 * carrier + 0.03 * dist, 0.7)))
        for phase, minutes in (("pickedUp", pick), ("delivered", deliv)):
            rows.append({"orderKey": i, "created": t0[i], "phase": phase, "carrier": carrier, "dist_km": dist,
                         "minutes": max(1.0, round(minutes))})
    path = tmp_path / "table.parquet"
    pd.DataFrame(rows).to_parquet(path)
    return path


@pytest.mark.parametrize("smearing", ["stratum", "none"])
def test_log_target_run_reports_hours(tmp_path, smearing):
    path = _table(tmp_path)
    user = {
        "run": {"name": f"log_{smearing}", "output_dir": str(tmp_path / "results"), "n_jobs": 1},
        "dataset": {"adapter": "generic", "path": str(path), "time_col": "created", "target_col": "minutes",
                    "target_scale": 1 / 60, "target_unit": "hours", "target_transform": "log",
                    "target_smearing": smearing, "group_col": "orderKey", "strata_col": "phase",
                    "categorical": ["carrier", "phase"]},
        "split": {"n_folds": 3},
        "tuning": {"max_rows": 1000, "early_stopping_rounds": 5, "max_estimators": 80,
                   "n_trials": {"LGBM": 2, "XGB": 1, "RIDGE": 1}},
        "base_learners": ["LGBM", "XGB", "RIDGE"],
        "stacking": {"top_k": 2, "meta_trials": 2, "oof_schemes": ["blocked"],
                     "meta_learners": ["EN_AVG", "EN_WAVG", "EN_LIN", "EN_LGBM"]},
        "statistics": {"bootstrap_resamples": 30},
        "robustness": {"seeds": [0]},
        "rolling": {"enabled": False},
        "sensitivity": {"enabled": True, "params": ["learning_rate"]},
        "explain": {"tree_shap_rows": 200, "pipeline_shap_rows": 5, "pipeline_permutations": 1,
                    "background_rows": 10},
        "profiling": {"enabled": False},
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(user))
    run_dir = run(load_config(cfg_path), resume=False)
    t = run_dir / "tables"
    test = pd.read_parquet(run_dir / "predictions" / "test.parquet")
    y = test["y"]
    hours = pd.read_parquet(path)["minutes"] / 60
    assert y.min() >= hours.min() - 1e-12 and y.max() <= hours.max() + 1e-12      # report scale is hours
    perf = pd.read_csv(t / "test_performance.csv").set_index("model")
    assert 0.3 < (test["LGBM"].mean() / y.mean()) < 3                               # predictions in hours too
    if smearing == "stratum":
        sm = pd.read_csv(t / "target_smearing.csv")
        assert {"pickedUp", "delivered", "_all_"} <= set(sm.stratum) and (sm.factor > 1).all()
    else:
        assert not (t / "target_smearing.csv").exists() or (pd.read_csv(t / "target_smearing.csv").factor == 1).all()
    # the deployable pipeline reproduces the reported predictions of the selected combiner, in hours
    pipe = joblib.load(run_dir / "models" / "pipeline.joblib")
    sample = pd.read_parquet(run_dir / "models" / "profiling_sample.parquet")
    assert np.allclose(pipe.predict(sample), test[pipe.name].to_numpy()[: len(sample)])
    if smearing == "none":                                                          # median retransformation
        assert np.allclose(pipe.predict(sample), np.exp(pipe.predict_model(sample)) / 60)
    else:                                                                           # bias-corrected upwards
        assert (pipe.predict(sample) > np.exp(pipe.predict_model(sample)) / 60).all()
    assert perf.loc[pipe.name, "RMSE"] > 0
    sens = pd.read_csv(t / "hyperparameter_sensitivity.csv")
    sel = sens[sens.parameter == "(selected)"].set_index("model")["test_RMSE"]
    for m in sel.index:
        assert sel[m] == pytest.approx(perf.loc[m, "RMSE"], rel=1e-9)            # same scale and correction
    # 1.4: per-stratum tables and grouping columns stored with the predictions
    bands = pd.read_csv(t / "duration_bands_by_stratum.csv")
    assert {"pickedUp", "delivered"} <= set(bands.stratum) and set(bands.band) <= {"short", "medium", "long"}
    dist = pd.read_csv(t / "target_distribution.csv")
    assert {"all", "pickedUp", "delivered"} <= set(dist.stratum)
    assert "phase" in test.columns or any(c.endswith("phase") for c in test.columns)
