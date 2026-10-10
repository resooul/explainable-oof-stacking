"""Tests of the changes introduced in version 1.4."""
import numpy as np
import pandas as pd
import pytest

from stackxai.cache import stable_section
from stackxai.config import DEFAULTS, deep_merge, load_config
from stackxai.models import make_learner, resolve_space
from stackxai.pipeline import rolling_learners
from stackxai.splits import inner_split
from stackxai.stacking import AverageMeta, StackedPipeline
from stackxai.tuning import sampler_seed


def grouped_starts(n_groups, size=2):
    return np.arange(0, n_groups * size, size), n_groups * size


def test_latest_validation_rows_are_the_most_recent_and_adjacent_to_fitting_rows():
    starts, n = grouped_starts(2000)
    train = np.arange(n)
    fit, val = inner_split(train, starts, 0.2, max_rows=500, position="latest")
    assert val.max() == n - 1                       # validation ends at the end of the segment
    assert fit.max() + 1 == val.min()               # fitting rows end where validation starts
    assert len(val) <= 100 + 2 and len(fit) <= 400 + 2
    assert fit.min() in set(starts) and val.min() in set(starts)


def test_latest_respects_a_separate_validation_cap_and_the_held_out_gap():
    starts, n = grouped_starts(2000)
    train = np.r_[np.arange(0, 1500), np.arange(2500, n)]     # interior fold: hole at 1500-2499
    fit, val = inner_split(train, starts, 0.2, max_rows=500, max_val_rows=300, position="latest")
    assert val.min() >= 2500 and fit.min() >= 2500           # drawn from the most recent contiguous run
    assert val.max() == n - 1 and fit.max() + 1 == val.min()
    assert len(val) <= 302


def test_positions_coincide_without_caps():
    starts, n = grouped_starts(1000)
    a = inner_split(np.arange(n), starts, 0.2, position="latest")
    b = inner_split(np.arange(n), starts, 0.2, position="after_cut")
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_after_cut_is_the_legacy_behaviour():
    starts, n = grouped_starts(2000)
    fit, val = inner_split(np.arange(n), starts, 0.2, max_rows=500)          # default position
    assert val.max() < n - 1                                                  # tail rows stay unused


def test_sampler_seed_per_split_differs_between_searches():
    t = {"sampler_seed": "per_split"}
    seeds = {sampler_seed(0, tag, t) for tag in ("blocked/final", "blocked/fold0", "blocked/fold1", "meta/EN_LGBM")}
    assert len(seeds) == 4
    assert sampler_seed(7, "blocked/fold0", {"sampler_seed": "shared"}) == 7
    assert sampler_seed(0, "x", t) == sampler_seed(0, "x", t)               # deterministic


def test_space_override_replaces_one_parameter():
    t = {"space_overrides": {"RIDGE": {"alpha": ["float_log", 0.001, 1.0e7]}}}
    assert resolve_space("RIDGE", t)["alpha"] == ("float_log", 0.001, 1.0e7)
    assert make_learner("RIDGE", ["x"], [], cfg=t).space["alpha"][2] == 1.0e7
    assert resolve_space("LGBM", t) == resolve_space("LGBM", {})
    with pytest.raises(ValueError):
        resolve_space("RIDGE", {"space_overrides": {"RIDGE": {"alpha": ["float", 5, 1]}}})


def test_legacy_values_keep_checkpoint_fingerprints():
    legacy = deep_merge(DEFAULTS, {"tuning": {"val_position": "after_cut", "sampler_seed": "shared",
                                              "space_overrides": {}}})
    section = stable_section(legacy, "tuning")
    assert not {"val_position", "sampler_seed", "space_overrides"} & set(section)
    new = stable_section(DEFAULTS, "tuning")
    assert new["val_position"] == "latest" and new["sampler_seed"] == "per_split"


def test_combiner_selection_must_be_a_computed_scheme(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("dataset: {path: x.csv, time_col: t, target_col: y}\n"
                    "stacking: {oof_schemes: [blocked], combiner_selection: forward}\n")
    with pytest.raises(ValueError, match="combiner_selection"):
        load_config(path)
    path.write_text("dataset: {path: x.csv, time_col: t, target_col: y}\nstacking: {oof_schemes: [blocked]}\n")
    assert load_config(path)["stacking"]["combiner_selection"] == "blocked"      # auto falls back
    path.write_text("dataset: {path: x.csv, time_col: t, target_col: y}\n")
    assert load_config(path)["stacking"]["combiner_selection"] == "forward"      # auto, both schemes
    path.write_text("dataset: {path: x.csv, time_col: t, target_col: y}\n"
                    "stacking: {oof_schemes: [blocked], combiner_selection: blocked}\n")
    assert load_config(path)["stacking"]["combiner_selection"] == "blocked"


def test_explicit_rolling_list_is_added_to_the_selected_families():
    assert rolling_learners(["LGBM", "XGB", "FTT"], ["CAT", "LGBM", "XGB"], []) == ["CAT", "LGBM", "XGB", "FTT"]
    assert rolling_learners("selected", ["CAT", "LGBM"], ["A"]) == ["CAT", "LGBM"]


def test_shared_encoding_gives_identical_predictions():
    rng = np.random.default_rng(0)
    n = 3000
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n),
                      "c": rng.choice(["p", "q", "r", "s"], n), "d": rng.integers(0, 50, n)})
    X.loc[rng.random(n) < 0.05, "a"] = np.nan
    y = X.a.fillna(0) * 2 + (X.c == "q") * 3 + np.sin(X.d) + rng.normal(size=n)
    feats, cats = ["a", "b", "c", "d"], ["c", "d"]
    params = {"LGBM": {"n_estimators": 60, "num_leaves": 15}, "XGB": {"n_estimators": 60, "max_depth": 4},
              "CAT": {"n_estimators": 60, "depth": 4}}
    bases = [make_learner(k, feats, cats, p, 0, 1, {}).fit(X, y) for k, p in params.items()]
    pipe = StackedPipeline(bases, AverageMeta(), feats)
    assert pipe.shares_encoding()
    X_new = X.iloc[:200].copy()
    X_new.loc[X_new.index[:10], "c"] = "unseen"                              # unseen categorical levels
    X_new.loc[X_new.index[10:20], "d"] = 999
    legacy = np.column_stack([b.predict(X_new) for b in bases])
    assert np.allclose(pipe.base_matrix(X_new), legacy, rtol=0, atol=1e-9)
    single = [pipe.base_matrix(X_new.iloc[[i]])[0] for i in range(5)]
    assert np.allclose(np.vstack(single), legacy[:5], rtol=0, atol=1e-9)


def test_pipeline_falls_back_when_a_learner_cannot_share_the_encoding():
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"a": rng.normal(size=500), "c": rng.choice(["p", "q"], 500)})
    y = X.a + (X.c == "q")
    bases = [make_learner("LGBM", ["a", "c"], ["c"], {"n_estimators": 20}, 0, 1, {}).fit(X, y),
             make_learner("RIDGE", ["a", "c"], ["c"], {}, 0, 1, {}).fit(X, y)]
    pipe = StackedPipeline(bases, AverageMeta(), ["a", "c"])
    assert not pipe.shares_encoding()
    assert np.allclose(pipe.base_matrix(X), np.column_stack([b.predict(X) for b in bases]))


def test_profiling_measures_the_shared_encoding_path(tmp_path):
    import joblib
    from stackxai.profiling import profile
    rng = np.random.default_rng(2)
    n = 1500
    X = pd.DataFrame({"a": rng.normal(size=n), "c": rng.choice(["p", "q", "r"], n)})
    y = X.a + (X.c == "q")
    bases = [make_learner(k, ["a", "c"], ["c"], {"n_estimators": 20}, 0, 1, {}).fit(X, y) for k in ("LGBM", "XGB", "CAT")]
    pipe = StackedPipeline(bases, AverageMeta(), ["a", "c"], name="EN_AVG")
    (tmp_path / "models").mkdir()
    (tmp_path / "tables").mkdir()
    joblib.dump(pipe, tmp_path / "models" / "pipeline.joblib")
    joblib.dump({"EN_AVG": AverageMeta()}, tmp_path / "models" / "meta_learners.joblib")
    X.iloc[:200].to_parquet(tmp_path / "models" / "profiling_sample.parquet")
    cfg = {"run": {"n_jobs": 1}, "profiling": {"latency_rows": 30, "warmup": 3, "workers": [1],
                                              "process_workers": [], "batch_rows": 200}}
    out = profile(tmp_path, cfg)
    stages = [s["stage"] for s in out["stages"]]
    assert stages == ["encode", "LGBM", "XGB", "CAT", "meta:EN_AVG", "end_to_end"]
    assert out["meta"][0]["implied_end_to_end_mean_ms"] > out["stages"][0]["mean_ms"]


def test_figure_labels_change_display_names_only(tmp_path):
    import shap
    from stackxai import explain
    ex = shap.Explanation(values=np.random.default_rng(0).normal(size=(30, 2)), base_values=np.zeros(30),
                          data=np.random.default_rng(1).normal(size=(30, 2)), feature_names=["hour", "dist_km"])
    shown = explain.relabel(ex, {"hour": "Order Hour"})
    assert shown.feature_names == ["Order Hour", "dist_km"] and ex.feature_names == ["hour", "dist_km"]
    explain.plot_explanation(ex, str(tmp_path / "f"), dependence_top=2, labels={"hour": "Order Hour"})
    assert (tmp_path / "f_dependence_hour.png").exists()                      # file names keep column names


def test_value_labels_rename_categorical_values_for_display():
    import shap
    from stackxai import explain
    from stackxai.reporting import model_label
    ex = shap.Explanation(values=np.zeros((3, 2)), base_values=np.zeros(3), data=np.array([[0, 1], [1, 2], [0, 3]]),
                          display_data=np.array([["delivered", 1], ["pickedUp", 2], ["delivered", 3]], dtype=object),
                          feature_names=["shipmentStatus", "productType"])
    shown = explain.relabel(ex, {"shipmentStatus": "Shipment Status"},
                            {"shipmentStatus": {"delivered": "Delivery", "pickedUp": "Pickup"}, "productType": {1: "Parcel"}})
    assert list(shown.display_data[:, 0]) == ["Delivery", "Pickup", "Delivery"]
    assert list(shown.display_data[:, 1]) == ["Parcel", 2, 3]                     # matched by text, others kept
    assert ex.display_data[0, 0] == "delivered"                                   # original untouched
    assert model_label("EN_AVG") == "EN-AVG" and model_label("FTT") == "FT-Transformer"


def test_value_labels_rename_displayed_categories():
    import shap
    from stackxai import explain
    ex = shap.Explanation(values=np.zeros((3, 2)), base_values=np.zeros(3), data=np.array([[0, 1.0], [1, 2.0], [0, 3.0]]),
                          display_data=np.array([["delivered", 1.0], ["pickedUp", 2.0], ["delivered", 3.0]], dtype=object),
                          feature_names=["shipmentStatus", "dist"])
    shown = explain.relabel(ex, {"shipmentStatus": "Target Status"},
                            {"shipmentStatus": {"pickedUp": "Pickup", "delivered": "Delivery"}})
    assert shown.feature_names[0] == "Target Status"
    assert list(shown.display_data[:, 0]) == ["Delivery", "Pickup", "Delivery"]
    assert list(ex.display_data[:, 0]) == ["delivered", "pickedUp", "delivered"]          # original unchanged


def test_category_dtype_with_unused_levels_gives_codes_within_the_vocabulary():
    """Parquet 'category' columns keep unused levels in their dtype; codes must still index the observed levels."""
    from stackxai.models import TabularEncoder
    rng = np.random.default_rng(3)
    n = 2000
    X = pd.DataFrame({"a": rng.normal(size=n),
                      "phase": pd.Categorical(rng.choice(["pickedUp", "delivered"], n),
                                              categories=["created", "delivered", "pickedUp", "returned"]),
                      "box": pd.Categorical(rng.choice([1, 2], n), categories=[1, 2, 3, 4, 5])})
    y = X.a + (X.phase == "delivered") * 2 + rng.normal(size=n) * 0.1
    enc = TabularEncoder(["a", "phase", "box"], ["phase", "box"]).fit(X)
    assert list(enc.categories["phase"]) == ["delivered", "pickedUp"] and list(enc.categories["box"]) == [1, 2]
    Xt = X.iloc[:50].copy()
    Xt.loc[Xt.index[:5], "phase"] = "returned"                  # level in the dtype but never observed
    Z = enc.transform(Xt)
    for c in ("phase", "box"):
        codes = Z[c].cat.codes.to_numpy()
        assert codes.max() < len(enc.categories[c]) and list(Z[c].cat.categories) == list(enc.categories[c])
    assert (Z["phase"].cat.codes.to_numpy()[:5] == -1).all()     # unseen -> missing
    bases = [make_learner(k, ["a", "phase", "box"], ["phase", "box"], p, 0, 1, {}).fit(X, y)
             for k, p in (("LGBM", {"n_estimators": 30}), ("XGB", {"n_estimators": 30}), ("CAT", {"n_estimators": 30}))]
    for k in ("RF", "RIDGE", "EBM"):
        assert np.isfinite(make_learner(k, ["a", "phase", "box"], ["phase", "box"], {}, 0, 1, {}).fit(X, y)
                           .predict(Xt)).all()
    pipe = StackedPipeline(bases, AverageMeta(), ["a", "phase", "box"])
    assert pipe.shares_encoding()
    assert np.allclose(pipe.base_matrix(Xt), np.column_stack([b.predict(Xt) for b in bases]))


def test_round_trip_of_integer_categories_with_missing_values():
    """Pipeline SHAP rebuilds rows from codes; integer categories with missing codes must be encoded again."""
    from stackxai.models import TabularEncoder
    rng = np.random.default_rng(4)
    n = 800
    X = pd.DataFrame({"a": rng.normal(size=n), "company": rng.integers(0, 40, n),
                      "phase": pd.Categorical(rng.choice(["p", "d"], n), categories=["d", "p", "x"]),
                      "box": pd.Categorical(rng.choice([1, 2, 3], n), categories=[1, 2, 3, 4])})
    y = X.a + (X.phase == "d") + X.company % 3
    cats, feats = ["company", "phase", "box"], list(X.columns)
    enc = TabularEncoder(feats, cats).fit(X)
    A = enc.to_numeric(X.iloc[:40])
    A[:5, 1] = np.nan
    A[5:10, 2] = np.nan
    A[10:15, 3] = np.nan
    back = enc.from_numeric(A)
    assert np.array_equal(np.isnan(enc.to_numeric(back)), np.isnan(A))
    assert np.allclose(np.nan_to_num(enc.to_numeric(back), nan=-1), np.nan_to_num(A, nan=-1))
    params = {"LGBM": {"n_estimators": 30}, "RIDGE": {}, "EBM": {"interactions": 0, "outer_bags": 2}}
    bases = [make_learner(k, feats, cats, p, 0, 1, {}).fit(X, y) for k, p in params.items()]
    pipe = StackedPipeline(bases, AverageMeta(), feats)
    assert not pipe.shares_encoding() and np.isfinite(pipe.base_matrix(back)).all()
