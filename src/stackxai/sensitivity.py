"""One-at-a-time hyperparameter sensitivity around the selected configuration.

For every selected base learner, each hyperparameter of its search space is moved
away from the value chosen for the final refit while all others are held fixed.
Every variant is processed exactly like the selected configuration: boosting
rounds or epochs are determined by early stopping on the same chronological inner
split of the training partition, the model is refitted on the full training
partition, and it is evaluated on the test partition. The analysis is descriptive:
it is run after the pipeline has been finalised and plays no role in any selection.
"""
from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd

from . import __cache_version__, metrics
from .cache import Cache, fingerprint, stable_section
from .models import make_learner, refit_params
from .splits import inner_split

log = logging.getLogger(__name__)


def variants(value, spec, log_factors, linear_step) -> list:
    """Values on both sides of ``value`` within the search range ``spec``."""
    kind, lo, hi, *step = spec
    if kind.endswith("_log"):
        cand = [value * f for f in log_factors]
    else:
        delta = linear_step * (hi - lo)
        if kind == "int":
            delta = max(delta, step[0] if step else 1)
        cand = [value - delta, value + delta]
    out = []
    for c in cand:
        c = float(np.clip(c, lo, hi))
        if kind.startswith("int"):
            q = step[0] if step else 1
            c = int(round((c - lo) / q) * q + lo)
        if not np.isclose(c, value) and c not in out:
            out.append(c)
    return out


def run_sensitivity(ds, res, cfg: dict, cache: Cache | None = None, scale=None) -> pd.DataFrame:
    """``scale`` (a pipeline.ReportScale) converts model-scale predictions to the report scale; every variant
    of a learner uses that learner's smearing factors, so variants differ only in their hyperparameters."""
    s, t = cfg["sensitivity"], cfg["tuning"]
    cache = cache or Cache(None)
    learners = res.selected if s["learners"] == "selected" else [n for n in s["learners"] if n in res.bases]
    X, y, train, test = ds.X, ds.y, res.train, res.test
    y_rep = ds.y_report
    if scale is None:
        from .pipeline import ReportScale
        scale = ReportScale(ds, res, cfg)
    fit_idx, val_idx = inner_split(train, ds.unit_starts(), t["val_fraction"], t.get("max_rows"),
                                   t.get("inner_split", "contiguous"), t.get("max_val_rows"),
                                   t.get("val_position", "after_cut"))
    seed, n_jobs = cfg["run"]["seed"], cfg["run"]["n_jobs"]
    rows_hash = fingerprint(train, test)
    out = []
    for name in learners:
        base = res.bases[name]
        probe = make_learner(name, ds.features, ds.categorical, cfg=t)
        space = probe.space
        selected = {k: v for k, v in base.final_params.items() if k != "n_estimators"}
        info = base.tuning[0] if base.tuning else {}
        test_sel = metrics.compute(y_rep[test], scale.convert(name, base.test, test))
        out.append({"model": name, "parameter": "(selected)", "value": np.nan, "selected_value": np.nan,
                    "inner_val_score": info.get("best_inner_score"), "n_estimators": base.final_params.get("n_estimators"),
                    **{f"test_{k}": test_sel[k] for k in ("RMSE", "MAE", "MAPE", "R2")}, "dRMSE_vs_selected": 0.0})
        params = s["params"] or list(space)
        for p in params:
            if p not in space or p not in selected:
                continue
            for v in variants(selected[p], space[p], s["log_factors"], s["linear_step"]):
                trial = {**selected, p: v}
                extra = [] if ds.transform.kind == "none" else [cfg["dataset"]["target_smearing"],
                                                                 scale.factors.get(name)]
                fp = fingerprint(__cache_version__, ds.fingerprint(), stable_section(cfg, "dataset"), cfg["split"],
                                 {k: v2 for k, v2 in stable_section(cfg, "tuning").items() if k != "n_trials"},
                                 name, trial, seed, rows_hash, *extra)

                def unit(trial=trial):
                    t0 = time.perf_counter()
                    lr = make_learner(name, ds.features, ds.categorical, trial, seed, n_jobs, t)
                    lr.fit(X.iloc[fit_idx], y[fit_idx], X.iloc[val_idx] if lr.iterative else None,
                           y[val_idx] if lr.iterative else None)
                    inner = metrics.score(t["metric"], y[val_idx], lr.predict(X.iloc[val_idx]))
                    final = refit_params(name, trial, lr.best_iteration, len(fit_idx), len(train), t)
                    model = make_learner(name, ds.features, ds.categorical, final, seed, n_jobs, t)
                    model.fit(X.iloc[train], y[train])
                    pred = scale.convert(name, model.predict(X.iloc[test]), test)
                    return inner, final.get("n_estimators"), metrics.compute(y_rep[test], pred), time.perf_counter() - t0

                inner, n_est, m, secs = cache.get(f"sensitivity/{name}/{p}={v}", unit, fp)
                log.info("sens | %-5s %-18s %-10s test_rmse=%.4f (%.0fs)", name, p, f"{v:.4g}", m["RMSE"], secs)
                out.append({"model": name, "parameter": p, "value": v, "selected_value": selected[p],
                            "inner_val_score": inner, "n_estimators": n_est,
                            **{f"test_{k}": m[k] for k in ("RMSE", "MAE", "MAPE", "R2")},
                            "dRMSE_vs_selected": m["RMSE"] - test_sel["RMSE"]})
    return pd.DataFrame(out)


def summarise(table: pd.DataFrame) -> pd.DataFrame:
    """Largest test-RMSE change per learner and parameter."""
    t = table[table.parameter != "(selected)"]
    if t.empty:
        return pd.DataFrame()
    g = t.groupby(["model", "parameter"])["dRMSE_vs_selected"]
    return pd.DataFrame({"min_dRMSE": g.min(), "max_dRMSE": g.max(),
                         "max_abs_dRMSE": g.apply(lambda x: x.abs().max())}).reset_index() \
        .sort_values(["model", "max_abs_dRMSE"], ascending=[True, False])
