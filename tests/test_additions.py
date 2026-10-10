"""Tests for the combiner-selection split, rolling reference, sensitivity variants and new report tables."""
import numpy as np
import pandas as pd
import pytest
import yaml

from stackxai.config import DEFAULTS, deep_merge, load_config
from stackxai.pipeline import rolling_reference, run
from stackxai.sensitivity import variants
from stackxai.stacking import run_meta


def _meta_data(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, 3))
    return Z, Z @ np.array([0.5, 0.3, 0.2]) + rng.normal(scale=0.1, size=n)


def _meta_cfg(mode, windows=4):
    return deep_merge(DEFAULTS, {"run": {"n_jobs": 1},
                                 "stacking": {"meta_trials": 2, "meta_selection": mode,
                                              "meta_selection_windows": windows},
                                 "tuning": {"early_stopping_rounds": 5, "max_estimators": 50}})


def test_holdout_selection_tunes_without_the_selection_rows():
    Z, y = _meta_data()
    cfg = _meta_cfg("holdout")
    res = run_meta("EN_LGBM", Z, y, Z[:10], ["A", "B", "C"], cfg, 0)
    cut = int(round(len(Z) * (1 - cfg["stacking"]["meta_val_fraction"])))
    assert len(res.selection_windows) == 1 and res.selection_windows[0]["start_row"] == cut
    assert res.tuning["inner_val_end"] < cut
    assert res.tuning["inner_fit_rows"] + res.tuning["inner_val_rows"] <= cut


@pytest.mark.parametrize("name", ["EN_LGBM", "EN_LIN", "EN_WAVG", "EN_AVG"])
def test_rolling_selection_scores_every_window_out_of_sample(name):
    Z, y = _meta_data()
    res = run_meta(name, Z, y, Z[:10], ["A", "B", "C"], _meta_cfg("rolling", 4), 0)
    w = res.selection_windows
    assert len(w) == 4
    assert all(a["end_row"] == b["start_row"] for a, b in zip(w, w[1:])) and w[-1]["end_row"] == len(Z)
    for r in w:
        assert r["fit_rows"] == r["start_row"]                     # fitted on earlier rows only
        if r["tuned_until_row"] is not None:
            assert r["tuned_until_row"] < r["start_row"]           # tuned on earlier rows only
    assert np.isclose(res.selection_score, np.mean([r["score"] for r in w]))


def test_selection_windows_respect_groups():
    from stackxai.stacking import selection_windows
    starts = np.arange(0, 1000, 4)                                 # groups of four rows
    wins = selection_windows(1000, starts, 4)
    assert len(wins) == 4 and wins[-1][1] == 1000
    assert all(lo % 4 == 0 for lo, _ in wins)


def test_max_val_rows_enlarges_only_the_validation_set():
    from stackxai.splits import inner_split
    starts = np.arange(10_000)
    fit0, val0 = inner_split(np.arange(10_000), starts, 0.2, max_rows=1000)
    fit1, val1 = inner_split(np.arange(10_000), starts, 0.2, max_rows=1000, max_val_rows=1500)
    assert len(fit0) == len(fit1) == 800 and np.array_equal(fit0, fit1)
    assert len(val0) == 200 and len(val1) == 1500 and val1[0] == fit1[-1] + 1


def test_new_neutral_settings_leave_checkpoint_fingerprints_unchanged():
    from stackxai.cache import stable_section
    cfg = deep_merge(DEFAULTS, {})
    # the 1.4 defaults (val_position: latest, sampler_seed: per_split) are new behaviour and enter fingerprints;
    # at the legacy values they are left out (see test_v14.test_legacy_values_keep_checkpoint_fingerprints)
    legacy_tuning = {k: v for k, v in cfg["tuning"].items() if k not in ("max_val_rows", "space_overrides")}
    legacy_dataset = {k: v for k, v in cfg["dataset"].items()
                      if k not in ("drop_features", "target_transform", "target_smearing")}
    assert stable_section(cfg, "tuning") == legacy_tuning
    assert stable_section(cfg, "dataset") == legacy_dataset
    changed = deep_merge(cfg, {"tuning": {"max_val_rows": 5000}})
    assert stable_section(changed, "tuning")["max_val_rows"] == 5000
    logged = deep_merge(cfg, {"dataset": {"target_transform": "log", "target_smearing": "global"}})
    assert stable_section(logged, "dataset")["target_transform"] == "log"      # changes what is trained
    assert "target_smearing" not in stable_section(logged, "dataset")         # only changes reporting


def test_shap_seed_model_follows_the_reference():
    from stackxai.pipeline import shap_seed_model
    assert shap_seed_model(["CAT", "LGBM", "XGB"]) == "CAT"
    assert shap_seed_model(["FTT", "XGB"]) == "XGB"
    assert shap_seed_model(["FTT", "MLP"]) is None


def test_rolling_reference_modes():
    assert rolling_reference("main", "LGBM", ["CAT", "LGBM"]) == "LGBM"
    assert rolling_reference("selected", "LGBM", ["CAT", "LGBM"]) == "CAT"
    assert rolling_reference("XGB", "LGBM", ["CAT", "LGBM"]) == "XGB"


def test_sensitivity_variants_stay_in_range():
    assert variants(0.1, ("float_log", 0.02, 0.2), [0.5, 2.0], 0.25) == [0.05, 0.2]
    assert variants(14, ("int", 4, 18), [0.5, 2.0], 0.25) == [10, 18]
    assert variants(1.0, ("float", 0.6, 1.0), [0.5, 2.0], 0.25) == [0.9]
    assert variants(88, ("int", 64, 256, 8), [0.5, 2.0], 0.25) == [64, 136]


def _long_table(n_orders=1500, seed=0):
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp("2021-01-01") + pd.to_timedelta(np.sort(rng.uniform(0, 200 * 86400, n_orders)), unit="s")
    carrier = rng.choice([f"c{i}" for i in range(12)], n_orders, p=np.r_[[0.3, 0.2, 0.15, 0.1, 0.1], [0.15 / 7] * 7])
    rows = []
    for i in range(n_orders):
        dist = rng.gamma(2, 3)
        for phase, base in (("pickup", 20), ("delivery", 60)):
            rows.append({"order_id": i, "created_at": t0[i], "phase": phase, "carrier": carrier[i],
                         "dist_km": dist, "items": int(rng.integers(1, 5)),
                         "duration_min": base + 4 * dist + 5 * (carrier[i] in ("c1", "c7")) + rng.gamma(2, 3)})
    df = pd.DataFrame(rows)
    df.loc[df.index[-60:], "carrier"] = "c_new"          # a level that occurs only in the test period
    return df


def test_generic_run_reports_new_tables(tmp_path):
    path = tmp_path / "table.parquet"
    _long_table().to_parquet(path)
    user = {
        "run": {"name": "additions", "output_dir": str(tmp_path / "results"), "n_jobs": 1},
        "dataset": {"adapter": "generic", "path": str(path), "time_col": "created_at", "target_col": "duration_min",
                    "group_col": "order_id", "strata_col": "phase", "categorical": ["carrier", "phase"],
                    "calendar": {"enabled": True}},
        "split": {"n_folds": 3},
        "tuning": {"max_rows": 800, "early_stopping_rounds": 5, "max_estimators": 60,
                   "n_trials": {"LGBM": 2, "XGB": 1, "DT": 1, "RIDGE": 1}},
        "base_learners": ["LGBM", "XGB", "DT", "RIDGE"],
        "stacking": {"top_k": 2, "meta_trials": 2, "meta_learners": ["EN_AVG", "EN_WAVG", "EN_LIN", "EN_LGBM"]},
        "statistics": {"bootstrap_resamples": 30, "extra_references": ["DT"]},
        "robustness": {"seeds": [0]},
        "rolling": {"n_origins": 2, "reference": "main"},
        "sensitivity": {"enabled": True, "params": ["learning_rate", "max_depth"]},
        "explain": {"enabled": False},
        "profiling": {"enabled": True, "latency_rows": 10, "warmup": 2, "workers": [1], "process_workers": [1],
                      "batch_rows": 200},
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(user))
    run_dir = run(load_config(cfg_path), resume=False)
    t = run_dir / "tables"
    for name in ("oof_by_fold", "performance_by_stratum", "stratum_sizes", "performance_by_category_frequency",
                 "rolling_comparison", "hyperparameter_sensitivity", "hyperparameter_sensitivity_summary",
                 "profiling_meta", "profiling_throughput"):
        assert (t / f"{name}.csv").exists(), name
    stratum = pd.read_csv(t / "performance_by_stratum.csv")
    assert set(stratum.stratum) == {"pickup", "delivery"}
    freq = pd.read_csv(t / "performance_by_category_frequency.csv")
    assert "unseen" in set(freq[freq.feature == "carrier"].level_frequency)
    tests = pd.read_csv(t / "statistical_tests.csv")
    assert {"DT"} <= set(tests.reference)
    assert {"MAE_direction", "RMSE_direction", "MAE_ci_excludes_zero", "RMSE_ci_excludes_zero"} <= set(tests.columns)
    assert set(tests.MAE_direction) <= {"lower error", "higher error", "inconclusive"}
    sel = pd.read_csv(t / "combiner_selection.csv")
    assert sel.selected.sum() == 1
    perf = pd.read_csv(t / "test_performance.csv")
    chosen = sel.loc[sel.selected, "combiner"].iloc[0]
    assert np.isclose(perf.set_index("model").loc[chosen, "validation_score"],
                      sel.set_index("combiner").loc[chosen, "selection_score"])
    assert (t / "combiner_selection_windows.csv").exists()
    hp = pd.read_csv(t / "hyperparameters.csv")
    assert hp.loc[hp.tag.str.startswith("blocked"), "inner_fit_contiguous"].astype(bool).all()
    roll = pd.read_csv(t / "rolling_origin.csv")
    ref = roll[roll.role == "reference"].model.unique()
    assert len(ref) == 1                                   # a fixed comparator across origins
    thr = pd.read_csv(t / "profiling_throughput.csv")
    assert thr["mode"].str.contains("processes").any()
    meta = pd.read_csv(t / "profiling_meta.csv")
    assert set(meta.combiner) == {"EN_AVG", "EN_WAVG", "EN_LIN", "EN_LGBM"}


def test_drop_features_removes_columns_and_rejects_unknown_names(tmp_path):
    from stackxai.data import load_dataset
    from synthetic import make_nyc_like
    csv = tmp_path / "train.csv"
    make_nyc_like(2000).to_csv(csv, index=False)
    base = deep_merge(DEFAULTS, {"dataset": {"adapter": "nyc_taxi", "path": str(csv)}})
    full = load_dataset(base)
    cut = load_dataset(deep_merge(base, {"dataset": {"drop_features": ["day_of_year", "month"]}}))
    assert set(full.features) - set(cut.features) == {"day_of_year", "month"}
    assert full.fingerprint() != cut.fingerprint()
    with pytest.raises(KeyError, match="drop_features"):
        load_dataset(deep_merge(base, {"dataset": {"drop_features": ["year"]}}))


def test_join_key_is_not_a_feature(tmp_path):
    from stackxai.data import load_dataset
    rng = np.random.default_rng(0)
    n = 400
    X = pd.DataFrame({"index": np.arange(n), "created": pd.date_range("2021-01-01", periods=n, freq="h"),
                      "a": rng.normal(size=n)})
    y = pd.DataFrame({"index": np.arange(n), "duration": rng.gamma(2, 3, n)})
    X.to_csv(tmp_path / "X.csv", index=False)
    y.to_csv(tmp_path / "y.csv", index=False)
    cfg = deep_merge(DEFAULTS, {"dataset": {"adapter": "generic", "path": str(tmp_path / "X.csv"),
                                            "target_path": str(tmp_path / "y.csv"), "join_key": "index",
                                            "time_col": "created", "target_col": "duration"}})
    ds = load_dataset(cfg)
    assert ds.features == ["a"]
