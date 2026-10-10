import importlib.util

import numpy as np
import pandas as pd
import pytest

from stackxai.config import DEFAULTS, deep_merge, load_config
from stackxai.models import make_learner
from stackxai.stats import paired_comparison

DEEP = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("rtdl_revisiting_models") is not None
LEARNERS = ["LGBM", "XGB", "CAT", "RF", "DT", "OLS", "RIDGE"] + (["MLP", "FTT", "TABNET"] if DEEP else [])


@pytest.mark.parametrize("name", LEARNERS)
def test_learners_handle_missing_and_unseen_categories(name):
    rng = np.random.default_rng(0)
    n = 400
    X = pd.DataFrame({"c": rng.choice(["a", "b", None], n), "x": rng.normal(size=n)})
    X.loc[::17, "x"] = np.nan
    y = X["x"].fillna(0) + (X["c"] == "a")
    cfg = {"max_estimators": 30, "early_stopping_rounds": 5, "deep": {"max_epochs": 2, "patience": 1}}
    params = {"n_estimators": 20} if name in {"LGBM", "XGB", "CAT"} else {}
    learner = make_learner(name, ["c", "x"], ["c"], params, 0, 1, cfg)
    learner.fit(X.iloc[:300], y[:300], X.iloc[300:], y[300:])
    pred = learner.predict(pd.DataFrame({"c": ["a", "unseen", None], "x": [0.0, np.nan, 1.0]}))
    assert pred.shape == (3,) and np.isfinite(pred).all()


def test_identical_predictions_do_not_break_the_tests():
    y = np.arange(20, dtype=float)
    out = paired_comparison(y, {"a": y + 1, "b": y + 1}, "a", n_resamples=10)
    assert out.loc[0, "p_value"] == 1.0


def test_inconsistent_band_configuration_is_rejected(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("evaluation:\n  band_quantiles: [0.5]\n  band_labels: [a, b, c]\n")
    with pytest.raises(ValueError, match="band_labels"):
        load_config(path)


def test_deep_merge_keeps_defaults():
    cfg = deep_merge(DEFAULTS, {"tuning": {"n_trials": {"LGBM": 1}}})
    assert cfg["tuning"]["n_trials"]["LGBM"] == 1
    assert cfg["tuning"]["n_trials"]["XGB"] == DEFAULTS["tuning"]["n_trials"]["XGB"]


def test_rolling_learner_specification():
    from stackxai.pipeline import rolling_learners
    assert rolling_learners("selected", ["CAT", "FTT"], ["LGBM", "CAT", "FTT"]) == ["CAT", "FTT"]
    assert rolling_learners("all", ["CAT"], ["LGBM", "CAT", "FTT"]) == ["LGBM", "CAT", "FTT"]
    assert rolling_learners(["XGB"], ["CAT"], ["LGBM"]) == ["CAT", "XGB"]     # explicit list adds to the selected


def test_unknown_model_names_fail_before_any_computation(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("base_learners: [LGBM, LGMB]\nstacking:\n  meta_learners: [EN_AVG, EN_FOO]\n")
    with pytest.raises(ValueError, match="LGMB.*EN_FOO"):
        load_config(path)


def test_unconstrained_linear_stacking_recovers_negative_weights():
    from stackxai.stacking import build_meta
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(500, 3))
    y = Z @ np.array([1.2, -0.3, 0.4]) + 2
    slr = build_meta("EN_SLR", list("abc"), {}, DEFAULTS, 0).fit(Z, y)
    lin = build_meta("EN_LIN", list("abc"), {}, DEFAULTS, 0).fit(Z, y)
    np.testing.assert_allclose(slr.weights, [1.2, -0.3, 0.4], atol=1e-6)
    assert (lin.weights >= 0).all()


def test_refit_rounds_scale_with_square_root_and_are_capped():
    from stackxai.models import refit_params
    t = {"max_estimators": 5000, "refit_scaling": "sqrt"}
    assert refit_params("LGBM", {}, 800, 100_000, 400_000, t)["n_estimators"] == 1600
    assert refit_params("XGB", {}, 800, 100_000, 400_000, {**t, "refit_scaling": "linear"})["n_estimators"] == 3200
    assert refit_params("CAT", {}, 800, 100_000, 400_000, {**t, "refit_scaling": "none"})["n_estimators"] == 800
    assert refit_params("LGBM", {}, 4000, 100_000, 400_000, t)["n_estimators"] == 5000
    assert refit_params("FTT", {}, 40, 100_000, 400_000, t)["n_estimators"] == 40
    assert refit_params("LGBM", {}, 800, 400_000, 100_000, t)["n_estimators"] == 800
