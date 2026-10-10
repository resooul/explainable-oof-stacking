"""Out-of-fold stacking with nested, leakage-free model selection."""
from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from . import __cache_version__, metrics
from .cache import Cache, fingerprint, stable_section
from .data import Dataset
from .models import Learner, make_learner, refit_params
from .target import TargetScale, to_report
from .splits import Fold, snap, temporal_folds
from .tuning import tune

log = logging.getLogger(__name__)
# Gradient-boosting combiners: name -> (learner, search space), and every combiner accepted by
# stacking.meta_learners.
META_TREES = {"EN_LGBM": ("LGBM", "META_LGBM"), "EN_XGB": ("XGB", "META_XGB"), "EN_CAT": ("CAT", "META_CAT")}
META_LEARNERS = ("EN_AVG", "EN_WAVG", "EN_LIN", "EN_SLR", *META_TREES)


@dataclass
class BaseResult:
    """Results of one base learner under one out-of-fold scheme: out-of-fold predictions on the training partition
    (NaN for rows without one), test predictions of the final refit, the hyperparameters of every fold and of the
    refit, tuning records, timings and, if kept, the refitted model.
    """
    name: str
    oof: np.ndarray
    test: np.ndarray
    fold_params: list[dict]
    final_params: dict
    tuning: list[dict]
    timing: dict
    model: Learner | None = None


@dataclass
class MetaResult:
    """Results of one combiner: test predictions, hyperparameters, selection score (mean over the selection windows)
    with its record per window, timings, the fitted combiner and its out-of-sample predictions on the selection
    windows.
    """
    name: str
    test: np.ndarray
    params: dict
    selection_score: float
    timing: dict
    model: object = None
    tuning: dict | None = None
    selection_windows: list[dict] = field(default_factory=list)
    oos: np.ndarray | None = None       # out-of-sample predictions on the selection windows (NaN elsewhere)


@dataclass
class StackResult:
    """Everything one out-of-fold scheme produces: partitions and folds, the results of every base learner and
    combiner, the selected base learners (the reference first), the training rows used as meta-rows, and the
    learners that failed.
    """
    scheme: str
    seed: int
    train: np.ndarray
    test: np.ndarray
    folds: list[Fold]
    bases: dict[str, BaseResult]
    selected: list[str]
    metas: dict[str, MetaResult] = field(default_factory=dict)
    meta_rows: np.ndarray | None = None
    failed: dict[str, str] = field(default_factory=dict)

    def test_predictions(self, clip_min=None) -> dict[str, np.ndarray]:
        """Test predictions of every base learner and combiner, optionally clipped from below at ``clip_min``."""
        out = {k: v.test for k, v in self.bases.items()} | {k: v.test for k, v in self.metas.items()}
        return {k: (np.maximum(v, clip_min) if clip_min is not None else v) for k, v in out.items()}

    def oof_matrix(self) -> np.ndarray:
        """Out-of-fold predictions of the selected base learners, one column each; these are the inputs of the
        combiners.
        """
        return np.column_stack([self.bases[b].oof for b in self.selected])


def _strip(params: dict) -> dict:
    """``params`` without the number of boosting rounds, which is set again for every fit."""
    return {k: v for k, v in params.items() if k != "n_estimators"}


def run_base(name: str, ds: Dataset, train: np.ndarray, test: np.ndarray, folds: list[Fold], cfg: dict,
             seed: int, mode: str, n_trials: int, scheme: str, keep_model: bool,
             fixed: tuple[list[dict], dict] | None = None, reuse_final: BaseResult | None = None,
             ckpt=None) -> BaseResult:
    """Tune, generate out-of-fold predictions and refit one base learner.

    Work is split into checkpointable units (final tuning, one unit per fold,
    final refit); ``ckpt(key, fn)`` returns a stored result or computes it.
    ``reuse_final`` supplies an already refitted full-training model, whose
    selection involves neither the folds nor the test rows, so that only the
    out-of-fold stage is recomputed for an alternative OOF scheme.
    """
    ckpt = ckpt or (lambda key, fn: fn())
    X, y, starts = ds.X, ds.y, ds.unit_starts()
    n_jobs = cfg["run"]["n_jobs"]
    records, timing = [], {"tuning_s": 0.0, "fold_fit_s": 0.0, "final_fit_s": 0.0, "predict_s": 0.0}
    common = dict(features=ds.features, categorical=ds.categorical, cfg=cfg, seed=seed, n_jobs=n_jobs)

    final_info = None
    if fixed is not None:
        final_params = fixed[1]
    elif reuse_final is not None:
        final_params, final_info = reuse_final.final_params, reuse_final.tuning[0]
    else:
        final_params, final_info = ckpt("final_tune", lambda: tune(
            name, X, y, train, starts, n_trials=0 if mode == "none" else n_trials, tag=f"{scheme}/final", **common))
        records.append(final_info)
        timing["tuning_s"] += final_info["tuning_seconds"]

    def fold_unit(i: int, f: Fold, warm: dict | None):
        """Hyperparameters, tuning record, predictions for the held-out rows and fit time of fold ``i``. Under
        nested tuning the fold has its own search, which under forward chaining starts from the previous fold's
        configuration (``tuning.warm_start``); otherwise the final hyperparameters are used with boosting rounds
        rescaled to the fold size.
        """
        info_f = None
        if fixed is not None:
            p = fixed[0][i]
        elif mode == "nested":
            use_warm = warm if (scheme == "forward" and cfg["tuning"]["warm_start"]) else None
            p, info_f = tune(name, X, y, f.train, starts, n_trials=n_trials, warm_start=use_warm,
                             tag=f"{scheme}/fold{i}", **common)
        else:
            p = refit_params(name, _strip(final_params), final_info["best_iteration"],
                             final_info["inner_fit_rows"], len(f.train), cfg["tuning"])
        t0 = time.perf_counter()
        learner = make_learner(name, ds.features, ds.categorical, p, seed, n_jobs, cfg["tuning"])
        learner.fit(X.iloc[f.train], y[f.train])
        fit_s = time.perf_counter() - t0
        return p, info_f, learner.predict(X.iloc[f.valid]), fit_s

    oof, fold_params, warm = np.full(len(train), np.nan), [], None
    for i, f in enumerate(folds):
        p, info_f, pred_f, fit_s = ckpt(f"fold{i}", lambda i=i, f=f, warm=warm: fold_unit(i, f, warm))
        if info_f is not None:
            records.append(info_f)
            timing["tuning_s"] += info_f["tuning_seconds"]
        timing["fold_fit_s"] += fit_s
        oof[np.searchsorted(train, f.valid)] = pred_f
        fold_params.append(p)
        warm = p

    def final_unit():
        """Fit the final model on the whole training partition and predict the test partition."""
        t0 = time.perf_counter()
        learner = make_learner(name, ds.features, ds.categorical, final_params, seed, n_jobs, cfg["tuning"])
        learner.fit(X.iloc[train], y[train])
        fit_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        pred = learner.predict(X.iloc[test])
        return learner, pred, fit_s, time.perf_counter() - t0

    if reuse_final is not None:
        final, pred = reuse_final.model, reuse_final.test
        timing["final_fit_s"], timing["predict_s"] = reuse_final.timing["final_fit_s"], reuse_final.timing["predict_s"]
    else:
        final, pred, timing["final_fit_s"], timing["predict_s"] = ckpt("final_fit", final_unit)
    log.info("base | %-6s %-8s seed=%d oof_rmse=%.4f fit=%.1fs", name, scheme, seed,
             metrics.rmse(y[train][np.isfinite(oof)], oof[np.isfinite(oof)]), timing["final_fit_s"])
    return BaseResult(name, oof, pred, fold_params, final_params, records, timing, final if keep_model else None)


def base_fingerprint(ds: Dataset, cfg: dict, name: str, scheme: str, seed: int, mode: str, n_trials: int,
                     train: np.ndarray, test: np.ndarray, fixed, reused: bool) -> str:
    """Every input that can change a base learner's result; unrelated settings are excluded."""
    tuning = {k: v for k, v in stable_section(cfg, "tuning").items() if k != "n_trials"}
    rows = [hashlib.sha1(np.ascontiguousarray(a, dtype=np.int64).tobytes()).hexdigest() for a in (train, test)]
    return fingerprint(__cache_version__, ds.fingerprint(), stable_section(cfg, "dataset"), cfg["split"], tuning,
                       name, scheme, seed, mode, n_trials, rows, fixed, reused)


def select_top_k(bases: dict[str, BaseResult], y_train: np.ndarray, metric: str, k: int) -> tuple[list[str], pd.DataFrame]:
    """Rank the candidates by their out-of-fold score on the training partition, using the rows for which every
    candidate has a prediction, and return the ``k`` best together with the full ranking."""
    mask = np.all(np.column_stack([np.isfinite(b.oof) for b in bases.values()]), axis=1)
    scores = {n: metrics.score(metric, y_train[mask], b.oof[mask]) for n, b in bases.items()}
    ranking = pd.Series(scores).sort_values()
    return list(ranking.index[:k]), ranking.rename("oof_score").to_frame()


# The combiners share one interface: fit(Z, y) on a matrix of base predictions, predict(Z), and weights, the
# learned linear weights (None for combiners without them).
class AverageMeta:
    """Simple average of the base predictions (EN_AVG)."""
    def fit(self, Z, y):
        return self

    def predict(self, Z):
        return np.asarray(Z).mean(axis=1)

    @property
    def weights(self):
        return None


class WeightedAverageMeta:
    """Convex combination (EN_WAVG): non-negative weights summing to one, optimised on the out-of-fold
    predictions for the loss ``stacking.wavg_objective``."""

    def __init__(self, objective="mse"):
        self.objective = objective
        self.w = None

    def fit(self, Z, y):
        Z, y = np.asarray(Z), np.asarray(y)
        losses = {"mse": lambda p: np.mean((y - p) ** 2), "mae": lambda p: np.mean(np.abs(y - p)),
                  "mape": lambda p: np.mean(np.abs((y - p) / np.where(y == 0, 1, y)))}
        loss = losses[self.objective]
        k = Z.shape[1]
        res = minimize(lambda w: loss(Z @ w), np.full(k, 1 / k), method="SLSQP", bounds=[(0, 1)] * k,
                       constraints=({"type": "eq", "fun": lambda w: w.sum() - 1},))
        self.w = np.clip(res.x, 0, None) / np.clip(res.x, 0, None).sum()
        return self

    def predict(self, Z):
        return np.asarray(Z) @ self.w

    @property
    def weights(self):
        return self.w


class LinearMeta:
    """Least-squares stacking with intercept: non-negative weights (EN_LIN; Breiman, 1996) or unconstrained
    weights (EN_SLR)."""

    def __init__(self, positive: bool = True):
        self.positive = positive

    def fit(self, Z, y):
        from sklearn.linear_model import LinearRegression
        self.model = LinearRegression(positive=self.positive).fit(np.asarray(Z), np.asarray(y))
        return self

    def predict(self, Z):
        return self.model.predict(np.asarray(Z))

    @property
    def weights(self):
        return self.model.coef_


class TreeMeta:
    """Gradient-boosting combiner (EN_LGBM, EN_XGB, EN_CAT) whose features are the base-learner predictions."""
    def __init__(self, learner: Learner):
        self.learner = learner

    def fit(self, Z, y):
        self.learner.fit(self._frame(Z), y)
        return self

    def predict(self, Z):
        return self.learner.predict(self._frame(Z))

    def _frame(self, Z):
        """Base predictions as a frame whose columns are named after the base learners."""
        return pd.DataFrame(np.asarray(Z), columns=self.learner.features)

    def contributions(self, Z):
        """TreeSHAP values of the base predictions and the base value."""
        return self.learner.contributions(self._frame(Z))

    def set_inference_threads(self, n):
        self.learner.set_inference_threads(n)

    @property
    def weights(self):
        return None


def build_meta(name: str, base_names: list[str], params: dict, cfg: dict, seed: int):
    """Instantiate the combiner ``name`` for the base learners ``base_names``; ``params`` holds the hyperparameters
    of a gradient-boosting combiner.
    """
    if name == "EN_AVG":
        return AverageMeta()
    if name == "EN_WAVG":
        return WeightedAverageMeta(cfg["stacking"]["wavg_objective"])
    if name == "EN_LIN":
        return LinearMeta(positive=True)
    if name == "EN_SLR":
        return LinearMeta(positive=False)
    if name in META_TREES:
        learner = make_learner(META_TREES[name][0], base_names, [], params, seed, cfg["run"]["n_jobs"], cfg["tuning"])
        learner.space_key = META_TREES[name][1]
        return TreeMeta(learner)
    raise KeyError(f"Unknown meta-learner '{name}'")


def selection_windows(n: int, starts: np.ndarray, k: int) -> list[tuple[int, int]]:
    """``k`` consecutive evaluation windows over the most recent ``k/(k+1)`` of ``n`` chronologically ordered
    meta-rows; window ``j`` is scored by a combiner fitted on every row before it. Boundaries respect groups."""
    bounds = sorted({snap(int(round(n * j / (k + 1))), starts, n) for j in range(1, k + 1)} | {n})
    out, lo = [], bounds[0]
    for hi in bounds[1:]:
        if hi > lo:
            out.append((lo, hi))
        lo = hi
    return [w for w in out if w[0] > 0]


def _tune_meta(name, Z, y, rows, starts, base_names, cfg, seed, tag):
    """Tune a tree meta-learner on ``rows`` only; returns the tuned parameters and the tuning record."""
    s = cfg["stacking"]
    family, space = META_TREES[name]
    meta_cfg = {**cfg, "tuning": {**cfg["tuning"], "val_fraction": s["meta_val_fraction"], "max_rows": None,
                                  "max_val_rows": None, "inner_split": "all"}}
    frame = pd.DataFrame(Z, columns=base_names)
    lo, hi = int(rows.min()), int(rows.max()) + 1
    tuned, info = tune(family, frame, y, rows, starts[(starts >= lo) & (starts < hi)], base_names, [], meta_cfg,
                       seed, s["meta_trials"], cfg["run"]["n_jobs"], space_key=space, tag=tag)
    return _strip(tuned), info


def _scaled(name, tuned, info, n_target, cfg):
    """Hyperparameters of a gradient-boosting combiner, with boosting rounds rescaled to ``n_target`` rows."""
    return refit_params(META_TREES[name][0], tuned, info.get("best_iteration"), info.get("inner_fit_rows", 1),
                        n_target, cfg["tuning"])


def run_meta(name: str, Z: np.ndarray, y: np.ndarray, Z_test: np.ndarray, base_names: list[str], cfg: dict,
             seed: int, fixed: dict | None = None, unit_starts: np.ndarray | None = None) -> MetaResult:
    """Fit one Level-1 combiner and score it for combiner selection.

    Meta-rows are in chronological order. ``stacking.meta_selection``:

    * ``rolling`` (default): the most recent ``k/(k+1)`` of the meta-dataset is cut into ``k``
      consecutive windows (``meta_selection_windows``); for each window the combiner is fitted
      (and, for tree meta-learners, tuned) on all earlier rows only and scored on the window. The
      selection score is the mean over windows, so the choice does not hinge on a single period.
    * ``holdout``: one chronological hold-out, the most recent ``meta_val_fraction``.

    In both modes no row used to score a combiner influences its fitting or tuning. The final
    combiner is then tuned (tree meta-learners) and fitted on the whole meta-dataset.
    Window boundaries respect group boundaries.
    """
    s = cfg["stacking"]
    metric = cfg["evaluation"]["primary_metric"]
    n = len(Z)
    starts = np.arange(n) if unit_starts is None else unit_starts
    tree = name in META_TREES and fixed is None
    t0 = time.perf_counter()
    if s["meta_selection"] == "rolling":
        windows = selection_windows(n, starts, s["meta_selection_windows"])
    else:
        cut = snap(int(round(n * (1 - s["meta_val_fraction"]))), starts, n)
        windows = [(cut, n)]
    records, tuned_last = [], None
    oos = np.full(n, np.nan)
    for j, (lo, hi) in enumerate(windows):
        fit_rows = np.arange(lo)
        params_j = dict(fixed or {})
        tuned_until = None
        if tree:
            tag = f"meta/{name}" if s["meta_selection"] == "holdout" else f"meta/{name}/sel{j}"
            tuned, info_j = _tune_meta(name, Z, y, fit_rows, starts, base_names, cfg, seed, tag)
            params_j, tuned_last = _scaled(name, tuned, info_j, lo, cfg), (tuned, info_j)
            tuned_until = info_j.get("inner_val_end")
        probe = build_meta(name, base_names, params_j, cfg, seed).fit(Z[fit_rows], y[fit_rows])
        oos[lo:hi] = probe.predict(Z[lo:hi])
        records.append({"window": j, "fit_rows": int(lo), "start_row": int(lo), "end_row": int(hi),
                        "n": int(hi - lo), "tuned_until_row": tuned_until,
                        "score": metrics.score(metric, y[lo:hi], oos[lo:hi])})
    sel = float(np.mean([r["score"] for r in records]))
    info = None
    params = dict(fixed or {})
    if tree and s["meta_selection"] == "holdout":
        # hold-out mode: the configuration tuned on the rows before the hold-out, rescaled to all meta-rows
        params, info = _scaled(name, *tuned_last, n, cfg), tuned_last[1]
    elif tree:
        tuned, info = _tune_meta(name, Z, y, np.arange(n), starts, base_names, cfg, seed, f"meta/{name}")
        params = _scaled(name, tuned, info, n, cfg)
    model = build_meta(name, base_names, params, cfg, seed).fit(Z, y)
    fit_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    pred = model.predict(Z_test)
    timing = {"meta_fit_s": fit_s, "meta_predict_s": time.perf_counter() - t1}
    return MetaResult(name, pred, params, sel, timing, model, info, records, oos)


def run_stack(ds: Dataset, train: np.ndarray, test: np.ndarray, cfg: dict, scheme: str, learners: list[str],
              seed: int, mode: str, trial_scale: float = 1.0, keep_models: bool = False,
              fixed: dict | None = None, selected: list[str] | None = None,
              reuse_final: dict[str, BaseResult] | None = None, cache: Cache | None = None,
              tag: str = "main", base_only: bool = False) -> StackResult:
    """Level-0 out-of-fold generation, top-k selection and Level-1 fitting for one OOF scheme.

    Base learners run in the configured order. With a ``cache``, every tuning,
    fold and refit unit is checkpointed, so an interrupted run resumes at the
    first unfinished unit and completed learners are skipped.

    ``fixed`` reuses the hyperparameters of another stack (see ``frozen_params``),
    ``selected`` replaces the top-k selection, ``reuse_final`` reuses the refitted
    models of another scheme, and ``base_only`` stops after the base learners.
    The combiners are fitted on the training rows for which every selected base
    learner has an out-of-fold prediction.
    """
    y = ds.y
    cache = cache or Cache(None)
    folds = temporal_folds(train, ds.unit_starts(), cfg["split"]["n_folds"], scheme, ds.times, cfg["split"]["gap"])
    trials = cfg["tuning"]["n_trials"]
    bases, failed = {}, {}
    for name in learners:
        fx = (fixed or {}).get("base", {}).get(name)
        n_tr = int(round(trials.get(name, 0) * trial_scale))
        reuse = (reuse_final or {}).get(name)
        fp = base_fingerprint(ds, cfg, name, scheme, seed, mode, n_tr, train, test, fx, reuse is not None)
        hits = cache.hits
        ckpt = lambda key, fn, name=name, fp=fp: cache.get(f"{tag}/{scheme}/seed{seed}/{name}/{key}", fn, fp)
        try:
            bases[name] = run_base(name, ds, train, test, folds, cfg, seed, mode, n_tr, scheme, keep_models,
                                   fx, reuse, ckpt)
        except Exception as exc:
            if not cfg["run"]["continue_on_error"]:
                raise
            failed[name] = f"{type(exc).__name__}: {exc}"
            log.exception("base | %s failed and is excluded (run.continue_on_error=true)", name)
            continue
        if cache.hits > hits:
            log.info("cache | %s/%s/seed%d/%s: %d unit(s) restored from checkpoints", tag, scheme, seed, name,
                     cache.hits - hits)
    learners = [n for n in learners if n in bases]
    if base_only:
        return StackResult(scheme, seed, train, test, folds, bases, [], failed=failed)
    y_train = y[train]
    if selected is None:
        selected, _ = select_top_k({k: bases[k] for k in learners}, y_train,
                                   cfg["stacking"]["selection_metric"], cfg["stacking"]["top_k"])
    res = StackResult(scheme, seed, train, test, folds, bases, list(selected), failed=failed)
    Z = res.oof_matrix()
    rows = np.flatnonzero(np.all(np.isfinite(Z), axis=1))
    res.meta_rows = rows
    Z_test = np.column_stack([bases[b].test for b in selected])
    meta_starts = np.flatnonzero(np.isin(train[rows], ds.unit_starts()))
    for m in cfg["stacking"]["meta_learners"]:
        fx = (fixed or {}).get("meta", {}).get(m)
        res.metas[m] = run_meta(m, Z[rows], y_train[rows], Z_test, list(selected), cfg, seed, fx, meta_starts)
        log.info("meta | %-8s %-8s seed=%d val_score=%.4f", m, scheme, seed, res.metas[m].selection_score)
    return res


def frozen_params(res: StackResult) -> dict:
    """Hyperparameters of every base learner (per fold and final) and of every combiner of ``res``, so that the
    stack can be refitted, e.g. with another seed, without new searches.
    """
    return {"base": {k: (b.fold_params, b.final_params) for k, b in res.bases.items()},
            "meta": {k: m.params for k, m in res.metas.items()}}


class StackedPipeline:
    """Deployable end-to-end predictor: raw features -> base learners -> combiner.

    ``predict`` returns report-scale predictions, retransformed and smearing-corrected (per stratum when
    ``strata_col`` is given); ``predict_model`` returns the model-scale output used for explanations.
    """

    def __init__(self, bases: list[Learner], meta, features: list[str], clip_min=None, name="",
                 transform: TargetScale | None = None, smearing: dict | None = None, strata_col: str | None = None):
        self.bases, self.meta, self.features, self.clip_min, self.name = bases, meta, features, clip_min, name
        self.transform, self.smearing, self.strata_col = transform or TargetScale(), smearing, strata_col

    def shares_encoding(self) -> bool:
        """True when every base learner applies the same encoding before predicting, so it can be done once."""
        bases = self.bases
        return (all(getattr(b, "shares_encoding", False) for b in bases)
                and all(bases[0].encoder.same_vocabulary(b.encoder) for b in bases[1:]))

    def encode(self, X: pd.DataFrame) -> pd.DataFrame:
        """Encoding shared by all base learners; valid when ``shares_encoding()`` is true."""
        return self.bases[0].encoder.transform(X)

    def base_matrix(self, X: pd.DataFrame) -> np.ndarray:
        """Predictions of every base learner for ``X``, one column each; rows are encoded once when the encoding is
        shared.
        """
        if self.shares_encoding():
            Z = self.encode(X)
            return np.column_stack([b.predict_encoded(Z) for b in self.bases])
        return np.column_stack([b.predict(X) for b in self.bases])

    def predict_model(self, X: pd.DataFrame) -> np.ndarray:
        """Prediction on the model scale (what the learners were trained on; used for explanations)."""
        return self.meta.predict(self.base_matrix(X))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Prediction on the report scale (``target_unit``), retransformed and bias-corrected."""
        ts = getattr(self, "transform", None) or TargetScale()
        col = getattr(self, "strata_col", None)
        strata = X[col].to_numpy(dtype=object) if col and col in X else None
        return to_report(self.predict_model(X), ts, getattr(self, "smearing", None), strata, self.clip_min)

    def set_inference_threads(self, n: int):
        """Number of prediction threads of every base learner and of the combiner."""
        for b in self.bases:
            b.set_inference_threads(n)
        if hasattr(self.meta, "set_inference_threads"):
            self.meta.set_inference_threads(n)
