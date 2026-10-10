import numpy as np
import pandas as pd
import pytest

from stackxai.config import DEFAULTS, deep_merge
from stackxai.data import geo_features, load_dataset
from stackxai.splits import given_partitions


def _table(n, start, seed, group_offset=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"t": pd.date_range(start, periods=n, freq="h"),
                         "order": np.repeat(np.arange(n // 2) + group_offset, 2),
                         "phase": np.tile(["pickup", "delivered"], n // 2),
                         "lat1": 24.7 + rng.normal(0, .05, n), "lon1": 46.7 + rng.normal(0, .05, n),
                         "lat2": 24.7 + rng.normal(0, .05, n), "lon2": 46.7 + rng.normal(0, .05, n),
                         "y": rng.gamma(2, 60, n)})


def _cfg(path, test_path=None):
    return deep_merge(DEFAULTS, {"dataset": {
        "adapter": "generic", "path": str(path), "test_path": str(test_path) if test_path else None,
        "time_col": "t", "target_col": "y", "group_col": "order", "strata_col": "phase", "categorical": ["phase"],
        "exclude": ["lat1", "lon1", "lat2", "lon2"],
        "geo": {"origin": ["lat1", "lon1"], "destination": ["lat2", "lon2"], "features": ["haversine_km"]}}})


def test_haversine_distance_and_missing_coordinates():
    df = pd.DataFrame({"a": [40.0, 0.0], "b": [-74.0, -74.0], "c": [41.0, 41.0], "d": [-74.0, -74.0]})
    out = geo_features(df, {"origin": ["a", "b"], "destination": ["c", "d"],
                            "features": ["haversine_km", "bearing_deg"]})
    assert out.loc[0, "haversine_km"] == pytest.approx(111.19, abs=0.05)
    assert out.loc[0, "bearing_deg"] == pytest.approx(0.0, abs=1e-6)
    assert np.isnan(out.loc[1, "haversine_km"])


def test_user_supplied_partitions_are_respected(tmp_path):
    _table(200, "2021-01-01", 0).to_csv(tmp_path / "train.csv", index=False)
    _table(80, "2021-02-01", 1, group_offset=1000).to_csv(tmp_path / "test.csv", index=False)
    ds = load_dataset(_cfg(tmp_path / "train.csv", tmp_path / "test.csv"))
    train, test, check = given_partitions(ds.frame["_partition"].to_numpy(), ds.times)
    assert (len(train), len(test)) == (200, 80) and check["chronological"]
    assert "haversine_km" in ds.features and "phase" in ds.features and "_partition" not in ds.features
    assert not set(ds.frame.order.iloc[train]) & set(ds.frame.order.iloc[test])


def test_group_shared_by_both_files_is_rejected(tmp_path):
    _table(200, "2021-01-01", 0).to_csv(tmp_path / "train.csv", index=False)
    _table(80, "2021-02-01", 1).to_csv(tmp_path / "test.csv", index=False)
    with pytest.raises(ValueError, match="occur in both files"):
        load_dataset(_cfg(tmp_path / "train.csv", tmp_path / "test.csv"))


def test_non_chronological_partitions_are_flagged(tmp_path):
    _table(200, "2021-01-01", 0).to_csv(tmp_path / "train.csv", index=False)
    _table(80, "2021-01-03", 1, group_offset=1000).to_csv(tmp_path / "test.csv", index=False)
    ds = load_dataset(_cfg(tmp_path / "train.csv", tmp_path / "test.csv"))
    _, _, check = given_partitions(ds.frame["_partition"].to_numpy(), ds.times)
    assert not check["chronological"] and check["test_rows_before_train_end"] > 0


def test_undeclared_text_column_is_rejected(tmp_path):
    _table(100, "2021-01-01", 0).assign(note="x").to_csv(tmp_path / "train.csv", index=False)
    with pytest.raises(TypeError, match="note"):
        load_dataset(_cfg(tmp_path / "train.csv"))


def test_derived_column_collision_is_rejected(tmp_path):
    _table(100, "2021-01-01", 0).assign(hour=1).to_csv(tmp_path / "train.csv", index=False)
    cfg = _cfg(tmp_path / "train.csv")
    cfg["dataset"]["calendar"] = {"enabled": True, "prefix": ""}
    with pytest.raises(ValueError, match="calendar"):
        load_dataset(cfg)


def _split_xy(frame, prefix, tmp_path, shuffle_y=False):
    X = frame.drop(columns="y")
    Y = frame[["order", "phase", "y"]].copy() if shuffle_y else frame[["y"]]
    if shuffle_y:
        Y["row"] = Y["order"].astype(str) + "_" + Y["phase"]
        X = X.assign(row=X["order"].astype(str) + "_" + X["phase"])
        Y = Y[["row", "y"]].sample(frac=1, random_state=0)
    X.to_csv(tmp_path / f"X_{prefix}.csv", index=False)
    Y.to_csv(tmp_path / f"y_{prefix}.csv", index=False)


def test_separate_feature_and_target_files(tmp_path):
    tr, te = _table(200, "2021-01-01", 0), _table(80, "2021-02-01", 1, group_offset=1000)
    _split_xy(tr, "train", tmp_path)
    _split_xy(te, "test", tmp_path)
    cfg = _cfg(tmp_path / "X_train.csv", tmp_path / "X_test.csv")
    cfg["dataset"].update(target_path=str(tmp_path / "y_train.csv"), test_target_path=str(tmp_path / "y_test.csv"))
    ds = load_dataset(cfg)
    train, test, _ = given_partitions(ds.frame["_partition"].to_numpy(), ds.times)
    assert (len(train), len(test)) == (200, 80)
    np.testing.assert_allclose(np.sort(ds.y[test]), np.sort(te["y"].to_numpy()))


def test_target_file_aligned_by_key_when_shuffled(tmp_path):
    tr = _table(200, "2021-01-01", 0)
    _split_xy(tr, "train", tmp_path, shuffle_y=True)
    cfg = _cfg(tmp_path / "X_train.csv")
    cfg["dataset"].update(target_path=str(tmp_path / "y_train.csv"), join_key="row",
                          exclude=cfg["dataset"]["exclude"] + ["row"])
    ds = load_dataset(cfg)
    expected = tr.set_index(tr["order"].astype(str) + "_" + tr["phase"])["y"]
    np.testing.assert_allclose(ds.y, expected.loc[ds.frame["row"]].to_numpy())


def test_misaligned_target_file_is_rejected(tmp_path):
    tr = _table(200, "2021-01-01", 0)
    tr.drop(columns="y").to_csv(tmp_path / "X.csv", index=False)
    tr[["y"]].iloc[:150].to_csv(tmp_path / "y.csv", index=False)
    cfg = _cfg(tmp_path / "X.csv")
    cfg["dataset"]["target_path"] = str(tmp_path / "y.csv")
    with pytest.raises(ValueError, match="rows but target file"):
        load_dataset(cfg)
