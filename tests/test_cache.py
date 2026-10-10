import numpy as np
import pandas as pd

from stackxai.cache import Cache
from stackxai.config import DEFAULTS, deep_merge
from stackxai.data import Dataset
from stackxai.splits import holdout
from stackxai.stacking import run_stack


def _setup():
    rng = np.random.default_rng(0)
    n = 2000
    frame = pd.DataFrame({"t": pd.date_range("2021-01-01", periods=n, freq="min"),
                          "x": rng.normal(size=n), "z": rng.normal(size=n)})
    frame["y"] = 2 * frame.x - frame.z + rng.normal(scale=0.5, size=n)
    ds = Dataset(frame=frame, features=["x", "z"], categorical=[], target="y", time_col="t")
    cfg = deep_merge(DEFAULTS, {"run": {"n_jobs": 1}, "split": {"n_folds": 3},
                                "tuning": {"n_trials": {"DT": 2, "RIDGE": 2}},
                                "stacking": {"top_k": 2, "meta_learners": ["EN_AVG", "EN_LIN"]}})
    return ds, cfg, *holdout(n, ds.unit_starts(), 0.3)


def test_completed_learners_are_restored_and_only_new_work_is_computed(tmp_path):
    ds, cfg, train, test = _setup()
    first = Cache(tmp_path)
    a = run_stack(ds, train, test, cfg, "blocked", ["DT", "RIDGE"], 0, "nested", cache=first)
    assert first.hits == 0 and first.misses == 2 * (3 + 2)

    second = Cache(tmp_path)
    b = run_stack(ds, train, test, cfg, "blocked", ["DT", "RIDGE", "OLS"], 0, "nested", cache=second)
    assert second.hits == 2 * 5 and second.misses == 5
    np.testing.assert_array_equal(a.bases["DT"].oof, b.bases["DT"].oof)

    changed = deep_merge(cfg, {"tuning": {"n_trials": {"DT": 3}}})
    third = Cache(tmp_path)
    run_stack(ds, train, test, changed, "blocked", ["DT", "RIDGE"], 0, "nested", cache=third)
    assert third.hits == 5 and third.misses == 5


def test_interrupted_learner_resumes_at_first_missing_unit(tmp_path):
    ds, cfg, train, test = _setup()
    run_stack(ds, train, test, cfg, "blocked", ["DT"], 0, "nested", cache=Cache(tmp_path))
    victim = next(tmp_path.rglob("fold2-*.joblib"))
    victim.unlink()
    resumed = Cache(tmp_path)
    run_stack(ds, train, test, cfg, "blocked", ["DT"], 0, "nested", cache=resumed)
    assert resumed.misses == 1 and resumed.hits == 4
    assert not list(tmp_path.rglob("*.tmp"))


def test_failed_learner_is_excluded_when_requested(tmp_path):
    ds, cfg, train, test = _setup()
    cfg = deep_merge(cfg, {"run": {"continue_on_error": True}})
    res = run_stack(ds, train, test, cfg, "blocked", ["DT", "RIDGE", "EBM_MISSING"], 0, "none")
    assert "EBM_MISSING" in res.failed and set(res.bases) == {"DT", "RIDGE"}
