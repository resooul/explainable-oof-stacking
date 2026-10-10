"""Dataset adapters and the common dataset container."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from .target import TargetScale

log = logging.getLogger(__name__)

# Dataset adapters by name (dataset.adapter); each takes the configuration and returns a Dataset.
ADAPTERS: dict[str, Callable[[dict], "Dataset"]] = {}


def register(name: str):
    """Decorator that registers a dataset adapter under ``name``, the value selected by ``dataset.adapter``."""
    def wrap(fn):
        """Store ``fn`` in ``ADAPTERS`` and return it unchanged."""
        ADAPTERS[name] = fn
        return fn
    return wrap


@dataclass
class Dataset:
    """Modelling table in temporal order, together with the roles of its columns.

    ``target`` names the target column on the report scale (``target_unit``); ``transform`` maps it to the model
    scale on which the learners are trained. ``group_col`` keeps correlated rows, such as the pickup and delivery
    rows of one order, on the same side of every split; ``strata_col`` defines subsets that are also reported
    separately; ``cluster_col`` defines the clusters resampled by the bootstrap. ``flow`` records how many rows each
    cleaning step removed.
    """
    frame: pd.DataFrame
    features: list[str]
    categorical: list[str]
    target: str
    time_col: str
    target_unit: str = "minutes"
    group_col: str | None = None
    strata_col: str | None = None
    cluster_col: str | None = None
    flow: list[dict] = field(default_factory=list)
    transform: TargetScale = field(default_factory=TargetScale)

    @property
    def X(self) -> pd.DataFrame:
        """Feature columns."""
        return self.frame[self.features]

    @property
    def y(self) -> np.ndarray:
        """Target on the model scale (what the learners are trained on)."""
        return self.transform.forward(self.y_report)

    @property
    def y_report(self) -> np.ndarray:
        """Target on the report scale (``target_unit``)."""
        return self.frame[self.target].to_numpy(dtype=float)

    def strata_values(self) -> np.ndarray | None:
        """Stratum of every row, or ``None`` when no stratum column is set."""
        return None if self.strata_col is None else self.frame[self.strata_col].to_numpy(dtype=object)

    @property
    def times(self) -> np.ndarray:
        """Timestamp of every row, i.e. the moment at which the prediction is made."""
        return self.frame[self.time_col].to_numpy()

    def fingerprint(self) -> list:
        """Content summary of the modelling table, used to invalidate checkpoints when the data change."""
        if not hasattr(self, "_fp"):
            y, t = self.y, self.times
            self._fp = [len(self.frame), self.features, self.categorical, float(np.nansum(y)),
                        float(np.nansum(y ** 2)), str(t.min()), str(t.max()), self.group_col]
        return self._fp

    def unit_starts(self) -> np.ndarray:
        """Row positions at which a new split unit (group or row) begins."""
        if self.group_col is None:
            return np.arange(len(self.frame))
        g = self.frame[self.group_col].to_numpy()
        change = np.r_[True, g[1:] != g[:-1]]
        return np.flatnonzero(change)


class FlowLog:
    """Records, for every cleaning step, the number of remaining rows and the number of rows the step removed."""
    def __init__(self):
        self.rows: list[dict] = []

    def __call__(self, step: str, frame: pd.DataFrame):
        prev = self.rows[-1]["rows"] if self.rows else len(frame)
        self.rows.append({"step": step, "rows": len(frame), "removed": prev - len(frame)})
        log.info("data | %-40s rows=%s", step, f"{len(frame):,}")


def temporal_order(frame: pd.DataFrame, time_col: str, group_col: str | None) -> pd.DataFrame:
    """Sort chronologically; rows of the same group are kept contiguous and ordered by the group's first time."""
    if group_col is None:
        return frame.sort_values(time_col, kind="mergesort").reset_index(drop=True)
    first = frame.groupby(group_col)[time_col].transform("min")
    return (frame.assign(_unit_time=first)
                 .sort_values(["_unit_time", group_col, time_col], kind="mergesort")
                 .drop(columns="_unit_time")
                 .reset_index(drop=True))


def _read_table(path: str | Path, nrows: int | None = None, usecols=None) -> pd.DataFrame:
    """Read a CSV or Parquet file, optionally limited to its first ``nrows`` rows and to the columns ``usecols``."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    if path.suffix == ".parquet":
        df = pd.read_parquet(path, columns=usecols)
        return df.head(nrows) if nrows else df
    return pd.read_csv(path, nrows=nrows, usecols=usecols)


def _add_cluster(frame: pd.DataFrame, cfg: dict, time_col: str) -> str | None:
    """Name of the column that defines bootstrap clusters, or ``None``. ``dataset.cluster: day`` derives a
    calendar-day column from ``time_col``, ``none`` disables clustering, and any other value names an existing
    column.
    """
    cluster = cfg.get("cluster", "day")
    if cluster in (None, "none"):
        return None
    if cluster == "day":
        frame["_cluster"] = frame[time_col].dt.strftime("%Y-%m-%d")
        return "_cluster"
    if cluster not in frame:
        raise KeyError(f"cluster column '{cluster}' not found")
    return cluster


def holiday_dates(spec, ts: pd.Series) -> pd.DatetimeIndex | None:
    """``None``, ``"us"`` (US federal calendar) or the path of a CSV file with a ``date`` column."""
    if spec in (None, "none"):
        return None
    if spec == "us":
        return USFederalHolidayCalendar().holidays(ts.min().floor("D"), ts.max().ceil("D"))
    return pd.DatetimeIndex(pd.to_datetime(_read_table(spec)["date"])).normalize()


def calendar_features(ts: pd.Series, prefix: str = "", holidays: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    """Calendar features of the timestamps ``ts``: hour, minute of day, weekday, day of month, day of year, ISO
    week, month and a weekend flag, plus a holiday flag when ``holidays`` is given. Column names start with
    ``prefix``.
    """
    out = pd.DataFrame({
        f"{prefix}hour": ts.dt.hour,
        f"{prefix}minute_of_day": ts.dt.hour * 60 + ts.dt.minute,
        f"{prefix}day_of_week": ts.dt.dayofweek,
        f"{prefix}day_of_month": ts.dt.day,
        f"{prefix}day_of_year": ts.dt.dayofyear,
        f"{prefix}week_of_year": ts.dt.isocalendar().week.astype(int).to_numpy(),
        f"{prefix}month": ts.dt.month,
        f"{prefix}is_weekend": (ts.dt.dayofweek >= 5).astype(int),
    }, index=ts.index)
    if holidays is not None:
        out[f"{prefix}is_holiday"] = ts.dt.normalize().isin(holidays).astype(int)
    return out


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in kilometres between two coordinate pairs given in degrees."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0088 * 2 * np.arcsin(np.sqrt(a))


def bearing_deg(lat1, lon1, lat2, lon2):
    """Initial bearing in degrees (0 to 360, clockwise from north) from the first to the second coordinate pair."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    x = np.sin(lon2 - lon1) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lon2 - lon1)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360


GEO_FEATURES = ("haversine_km", "manhattan_km", "bearing_deg")


def geo_features(df: pd.DataFrame, geo: dict) -> pd.DataFrame:
    """Great-circle distance, Manhattan approximation and bearing between two coordinate pairs.

    ``geo = {"origin": [lat, lon], "destination": [lat, lon], "features": [...], "zero_as_missing": true}``.
    Missing coordinates yield missing features.
    """
    (la1, lo1), (la2, lo2) = geo["origin"], geo["destination"]
    A = df[[la1, lo1, la2, lo2]].astype(float)
    if geo.get("zero_as_missing", True):
        A = A.mask(A == 0)
    a = A.to_numpy().T
    out = {"haversine_km": haversine_km(*a),
           "manhattan_km": haversine_km(a[0], a[1], a[0], a[3]) + haversine_km(a[0], a[3], a[2], a[3]),
           "bearing_deg": bearing_deg(*a)}
    names = geo.get("features") or ["haversine_km"]
    unknown = set(names) - set(GEO_FEATURES)
    if unknown:
        raise KeyError(f"Unknown geo features {sorted(unknown)}; available: {GEO_FEATURES}")
    return pd.DataFrame({n: out[n] for n in names}, index=df.index)


@register("nyc_taxi")
def load_nyc_taxi(cfg: dict) -> Dataset:
    """Adapter for the training file (train.csv) of the Kaggle 'New York City Taxi Trip Duration' competition.

    The cleaning rules of ``dataset.cleaning`` are applied and logged step by step; distance, bearing and calendar
    features are derived from the pickup record, and the trip duration in minutes is the target.
    """
    c = cfg["dataset"]
    rules = {"min_duration_s": 60, "max_duration_s": 18_000,
             "lat_range": [40.49, 40.92], "lon_range": [-74.27, -73.68],
             "passenger_range": [1, 6], "min_distance_km": 0.05, **(c.get("cleaning") or {})}
    flow = FlowLog()
    df = _read_table(c["path"], nrows=c.get("sample_rows"))
    flow("raw", df)
    df = df.drop_duplicates("id")
    flow("duplicate trip ids removed", df)
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    df["dropoff_datetime"] = pd.to_datetime(df["dropoff_datetime"])
    elapsed = (df["dropoff_datetime"] - df["pickup_datetime"]).dt.total_seconds()
    df = df[np.isclose(elapsed, df["trip_duration"], atol=1)]
    flow("timestamps inconsistent with duration", df)
    df = df[df["trip_duration"].between(rules["min_duration_s"], rules["max_duration_s"])]
    flow("duration outside bounds", df)
    lat, lon = rules["lat_range"], rules["lon_range"]
    inside = (df["pickup_latitude"].between(*lat) & df["dropoff_latitude"].between(*lat)
              & df["pickup_longitude"].between(*lon) & df["dropoff_longitude"].between(*lon))
    df = df[inside]
    flow("coordinates outside bounding box", df)
    df = df[df["passenger_count"].between(*rules["passenger_range"])]
    flow("passenger count outside range", df)
    args = df[["pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"]].to_numpy().T
    df["haversine_km"] = haversine_km(*args)
    df = df[df["haversine_km"] >= rules["min_distance_km"]].copy()
    flow("distance below minimum", df)

    args = df[["pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude"]].to_numpy().T
    df["manhattan_km"] = (haversine_km(args[0], args[1], args[0], args[3])
                          + haversine_km(args[0], args[3], args[2], args[3]))
    df["bearing_deg"] = bearing_deg(*args)
    df["store_and_fwd_flag"] = (df["store_and_fwd_flag"] == "Y").astype(int)
    df = pd.concat([df, calendar_features(df["pickup_datetime"], holidays=holiday_dates("us", df["pickup_datetime"]))],
                   axis=1)
    df["duration_min"] = df["trip_duration"] / 60.0

    features = ["vendor_id", "passenger_count", "store_and_fwd_flag",
                "pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude",
                "haversine_km", "manhattan_km", "bearing_deg",
                "hour", "minute_of_day", "day_of_week", "day_of_month", "day_of_year",
                "week_of_year", "month", "is_weekend", "is_holiday"]
    ext = c.get("external_features")
    if ext:
        extra = _read_table(ext["path"], usecols=[ext["key"], *ext["columns"]])
        df = df.merge(extra, on=ext["key"], how="left")
        features += list(ext["columns"])
        flow("external features merged", df)

    df = temporal_order(df, "pickup_datetime", None)
    cluster = _add_cluster(df, c, "pickup_datetime")
    return Dataset(frame=df, features=features, categorical=["vendor_id"], target="duration_min",
                   time_col="pickup_datetime", target_unit="minutes", cluster_col=cluster,
                   flow=flow.rows)


def _attach_target(X: pd.DataFrame, Y: pd.DataFrame, c: dict) -> pd.DataFrame:
    """Combine a feature file with a separate target file, by ``join_key`` or, if absent, by row order."""
    target, key = c["target_col"], c.get("join_key")
    if target not in Y:
        if Y.shape[1] - (key in Y) != 1:
            raise KeyError(f"Target file must contain '{target}' or exactly one value column")
        Y = Y.rename(columns={next(col for col in Y.columns if col != key): target})
    if target in X:
        raise ValueError(f"'{target}' occurs in both the feature and the target file")
    if key:
        for name, frame in (("feature", X), ("target", Y)):
            if key not in frame:
                raise KeyError(f"join_key '{key}' missing from the {name} file")
            if frame[key].duplicated().any():
                raise ValueError(f"join_key '{key}' is not unique in the {name} file")
        out = X.merge(Y[[key, target]], on=key, how="left", validate="one_to_one")
        if out[target].isna().any() or len(Y) != len(X):
            raise ValueError(f"Feature and target files do not match one-to-one on '{key}'")
        return out
    if len(X) != len(Y):
        raise ValueError(f"Feature file has {len(X):,} rows but target file has {len(Y):,}; "
                         "set dataset.join_key to align them by an identifier")
    return X.assign(**{target: Y[target].to_numpy()})


@register("generic")
def load_generic(cfg: dict) -> Dataset:
    """Adapter for any pre-engineered table with a timestamp, a target and feature columns.

    Every column other than the time, target, group, join-key and excluded columns becomes a feature unless
    ``dataset.features`` lists them explicitly. Optional geo and calendar features are derived before that choice.
    """
    c = cfg["dataset"]
    for key in ("path", "time_col", "target_col"):
        if not c.get(key):
            raise ValueError(f"dataset.{key} is required for the generic adapter")
    flow = FlowLog()
    df = _read_table(c["path"], nrows=c.get("sample_rows"))
    if c.get("target_path"):
        df = _attach_target(df, _read_table(c["target_path"], nrows=c.get("sample_rows")), c)
    flow("raw", df)
    df[c["time_col"]] = pd.to_datetime(df[c["time_col"]])
    df = df[df[c["time_col"]].notna()]
    flow("missing timestamp removed", df)
    df["_target"] = df[c["target_col"]].astype(float) * float(c.get("target_scale", 1.0))
    df = df[np.isfinite(df["_target"])]
    flow("non-finite target removed", df)
    def append(frame, derived, source):
        """Append the ``derived`` columns to ``frame``; columns that already exist are never overwritten."""
        clash = sorted(set(derived.columns) & set(frame.columns))
        if clash:
            raise ValueError(f"Columns derived by dataset.{source} already exist in the data: {clash}")
        return pd.concat([frame, derived], axis=1)

    geo = c.get("geo")
    if geo:
        df = append(df, geo_features(df, geo), "geo")
    cal = c.get("calendar") or {}
    if cal.get("enabled"):
        weekend = cal.get("weekend_days")
        feats = calendar_features(df[c["time_col"]], cal.get("prefix", "cal_"),
                                  holiday_dates(cal.get("holidays"), df[c["time_col"]]))
        if weekend is not None:
            feats[f"{cal.get('prefix', 'cal_')}is_weekend"] = df[c["time_col"]].dt.dayofweek.isin(weekend).astype(int)
        df = append(df, feats, "calendar")
    # identifiers never enter the feature set: a row key that follows the row order would act as a trend feature
    reserved = {c["time_col"], c["target_col"], "_target", c.get("group_col"), c.get("join_key"),
                *c.get("exclude", [])} - {None}
    features = c.get("features") or [col for col in df.columns if col not in reserved]
    df = temporal_order(df, c["time_col"], c.get("group_col"))
    cluster = _add_cluster(df, c, c["time_col"])
    return Dataset(frame=df, features=list(features), categorical=list(c.get("categorical", [])),
                   target="_target", time_col=c["time_col"], target_unit=c.get("target_unit", "minutes"),
                   group_col=c.get("group_col"), strata_col=c.get("strata_col"),
                   cluster_col=cluster, flow=flow.rows)


def _with_partitions(cfg: dict, adapter) -> Dataset:
    """Load user-supplied training and test files with identical cleaning and feature engineering."""
    c = cfg["dataset"]
    parts = {}
    if bool(c.get("target_path")) != bool(c.get("test_target_path")):
        raise ValueError("dataset.target_path and dataset.test_target_path must be given together")
    for label, path, tpath in (("train", c["path"], c.get("target_path")),
                               ("test", c["test_path"], c.get("test_target_path"))):
        parts[label] = adapter({**cfg, "dataset": {**c, "path": path, "target_path": tpath}})
    tr, te = parts["train"], parts["test"]
    if tr.features != te.features:
        raise ValueError("Training and test files produced different feature sets")
    if tr.group_col:
        shared = set(tr.frame[tr.group_col]) & set(te.frame[te.group_col])
        if shared:
            raise ValueError(f"{len(shared):,} groups ('{tr.group_col}') occur in both files; "
                             "a group must belong to exactly one partition")
    frame = pd.concat([tr.frame.assign(_partition="train"), te.frame.assign(_partition="test")], ignore_index=True)
    frame = temporal_order(frame, tr.time_col, tr.group_col)
    flow = [{**r, "step": f"train: {r['step']}"} for r in tr.flow] + [{**r, "step": f"test: {r['step']}"} for r in te.flow]
    return Dataset(frame=frame, features=tr.features, categorical=tr.categorical, target=tr.target,
                   time_col=tr.time_col, target_unit=tr.target_unit, group_col=tr.group_col,
                   strata_col=tr.strata_col, cluster_col=tr.cluster_col, flow=flow)


def load_dataset(cfg: dict) -> Dataset:
    """Load the dataset named by ``dataset.adapter`` and prepare it for the pipeline.

    With ``dataset.test_path`` the training and test files are loaded separately and combined. The target
    transformation is attached, ``dataset.drop_features`` are removed, and every feature is checked to exist and to
    be numeric or declared categorical.
    """
    name = cfg["dataset"]["adapter"]
    if name not in ADAPTERS:
        raise KeyError(f"Unknown dataset adapter '{name}'. Available: {sorted(ADAPTERS)}")
    ds = _with_partitions(cfg, ADAPTERS[name]) if cfg["dataset"].get("test_path") else ADAPTERS[name](cfg)
    kind = cfg["dataset"].get("target_transform", "none")
    ds.transform = TargetScale(kind, float(cfg["dataset"].get("target_scale", 1.0)))
    if kind == "log" and (ds.y_report <= 0).any():
        raise ValueError(f"target_transform 'log' needs a positive target; {(ds.y_report <= 0).sum():,} rows are "
                         "zero or negative")
    drop = list(cfg["dataset"].get("drop_features") or [])
    if drop:
        unknown = [f for f in drop if f not in ds.features]
        if unknown:
            raise KeyError(f"dataset.drop_features lists columns that are not features: {unknown}. "
                           f"Features: {ds.features}")
        ds.features = [f for f in ds.features if f not in drop]
        ds.categorical = [f for f in ds.categorical if f not in drop]
    missing = [f for f in ds.features if f not in ds.frame]
    if missing:
        raise KeyError(f"Features missing from the data: {missing}")
    for col in (ds.time_col, ds.target, ds.group_col):
        if col in ds.features:
            raise ValueError(f"Column '{col}' is the time, target or group column and cannot be a feature")
    bad = [f for f in ds.features if f not in ds.categorical and not (
        pd.api.types.is_numeric_dtype(ds.frame[f]) or pd.api.types.is_bool_dtype(ds.frame[f]))]
    if bad:
        raise TypeError(f"Non-numeric features must be declared in dataset.categorical or listed in "
                        f"dataset.exclude: {bad}")
    log.info("data | %d features (%d categorical): %s", len(ds.features), len(ds.categorical), ", ".join(ds.features))
    return ds
