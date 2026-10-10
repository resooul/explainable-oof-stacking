"""Budgeted hyperparameter optimisation on a chronological inner split."""
from __future__ import annotations

import logging
import time
import zlib

import numpy as np
import optuna
import pandas as pd

from . import metrics
from .models import make_learner, refit_params, sample_space
from .splits import inner_split

log = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def sampler_seed(seed: int, tag: str, tuning: dict) -> int:
    """Seed of the TPE sampler. ``tuning.sampler_seed: per_split`` derives a distinct seed for every search
    (out-of-fold split, final refit, combiner window) from its tag, so that the random start-up configurations
    differ between searches; ``shared`` uses the run seed for every search."""
    mode = tuning.get("sampler_seed", "shared")
    if mode == "shared":
        return seed
    if mode != "per_split":
        raise ValueError(f"tuning.sampler_seed must be 'shared' or 'per_split', got '{mode}'")
    return seed + zlib.crc32(tag.encode()) % 100_000


def tune(name: str, X: pd.DataFrame, y: np.ndarray, train: np.ndarray, unit_starts: np.ndarray,
         features, categorical, cfg: dict, seed: int, n_trials: int, n_jobs: int,
         space_key: str | None = None, warm_start: dict | None = None, tag: str = "") -> tuple[dict, dict]:
    """Select the hyperparameters of learner ``name`` using only the rows in ``train``.

    The rows are split chronologically by ``splits.inner_split`` into fitting and validation rows (settings
    ``tuning.inner_split``, ``val_position``, ``val_fraction``, ``max_rows`` and ``max_val_rows``). Every
    configuration is fitted on the fitting rows and scored on the validation rows with ``tuning.metric``;
    iterative learners stop early on the validation rows. ``warm_start`` is evaluated as the first trial.
    Returns refit-ready parameters, with boosting rounds rescaled to ``len(train)``, and a record of the search.
    """
    t = cfg["tuning"]
    probe = make_learner(name, features, categorical, cfg=t)
    if space_key:
        probe.space_key = space_key
    space = probe.space
    fit_idx, val_idx = inner_split(train, unit_starts, t["val_fraction"], t.get("max_rows"),
                                   t.get("inner_split", "contiguous"), t.get("max_val_rows"),
                                   t.get("val_position", "after_cut"))
    Xf, yf, Xv, yv = X.iloc[fit_idx], y[fit_idx], X.iloc[val_idx], y[val_idx]
    t0 = time.perf_counter()

    def evaluate(params):
        """Fit one configuration on the fitting rows; return its validation score and early-stopped iteration count."""
        learner = make_learner(name, features, categorical, params, seed, n_jobs, t)
        learner.fit(Xf, yf, Xv if learner.iterative else None, yv if learner.iterative else None)
        return metrics.score(t["metric"], yv, learner.predict(Xv)), learner.best_iteration

    if not space or n_trials <= 0:
        best_params, best_iter, best_score, n_done = {}, None, None, 0
        if probe.iterative:
            best_score, best_iter = evaluate({})
    else:
        study = optuna.create_study(direction="minimize",
                                    sampler=optuna.samplers.TPESampler(seed=sampler_seed(seed, tag, t),
                                                                       n_startup_trials=max(3, n_trials // 3)))
        if warm_start:
            study.enqueue_trial({k: v for k, v in warm_start.items() if k in space})

        def objective(trial):
            """Optuna objective; the early-stopped iteration count is stored with the trial."""
            s, it = evaluate(sample_space(trial, space))
            trial.set_user_attr("best_iteration", it)
            return s

        study.optimize(objective, n_trials=n_trials, timeout=t.get("timeout_s"))
        best_params = sample_space(optuna.trial.FixedTrial(study.best_params), space)
        best_iter = study.best_trial.user_attrs.get("best_iteration")
        best_score, n_done = study.best_value, len(study.trials)

    params = refit_params(name, best_params, best_iter, len(fit_idx), len(train), t)
    info = {"model": name, "tag": tag, "n_trials": n_done, "inner_fit_rows": len(fit_idx),
            "inner_val_rows": len(val_idx), "sampler_seed": sampler_seed(seed, tag, t), "inner_fit_start": int(fit_idx.min()), "inner_fit_end": int(fit_idx.max()),
            "inner_val_start": int(val_idx.min()), "inner_val_end": int(val_idx.max()),
            "inner_fit_contiguous": bool(np.all(np.diff(fit_idx) == 1)), "best_inner_score": best_score, "best_iteration": best_iter,
            "tuning_seconds": time.perf_counter() - t0, **{f"param.{k}": v for k, v in params.items()}}
    log.info("tune | %-5s %-18s trials=%-3d score=%s  %.1fs", name, tag, n_done,
             "n/a" if best_score is None else f"{best_score:.4f}", info["tuning_seconds"])
    return params, info
