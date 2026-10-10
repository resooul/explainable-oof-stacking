"""Configuration loading with defaults and deep merging."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import yaml

from .target import SMEARING, TRANSFORMS

DEFAULTS: dict = {
    "run": {"name": "experiment", "output_dir": "results", "seed": 0, "n_jobs": -1, "continue_on_error": False,
            "stop_after": None},
    "dataset": {
        "adapter": "generic",
        "path": None,
        "test_path": None,
        "target_path": None,
        "test_target_path": None,
        "join_key": None,
        "sample_rows": None,
        "time_col": None,
        "target_col": None,
        "target_scale": 1.0,
        "target_unit": "minutes",
        "features": None,
        "drop_features": [],
        "target_transform": "none",
        "target_smearing": "stratum",
        "categorical": [],
        "exclude": [],
        "group_col": None,
        "strata_col": None,
        "cluster": "day",
        "cleaning": {},
        "external_features": None,
        "calendar": None,
        "geo": None,
    },
    "split": {"test_fraction": 0.3, "n_folds": 5, "gap": None},
    "tuning": {
        "mode": "nested",
        "metric": "rmse",
        "val_fraction": 0.2,
        "max_rows": 150_000,
        "max_val_rows": None,
        "inner_split": "contiguous",
        "val_position": "latest",
        "sampler_seed": "per_split",
        "space_overrides": {},
        "early_stopping_rounds": 100,
        "max_estimators": 10_000,
        "refit_scaling": "sqrt",
        "timeout_s": None,
        "warm_start": True,
        "n_trials": {"LGBM": 20, "XGB": 20, "CAT": 20, "RF": 20, "DT": 20, "RIDGE": 20, "EBM": 20,
                     "MLP": 20, "RESNET": 20, "CNN": 20, "LSTM": 20, "FTT": 20, "TABNET": 20},
        "deep": {"device": "auto", "batch_size": 1024, "eval_batch_size": 8192, "max_epochs": 100, "patience": 8},
    },
    "base_learners": ["LGBM", "XGB", "CAT", "RF", "DT", "OLS", "RIDGE"],
    "stacking": {
        "top_k": 3,
        "selection_metric": "rmse",
        "oof_schemes": ["blocked", "forward"],
        "primary_scheme": "blocked",
        "secondary_learners": "selected",
        "meta_learners": ["EN_AVG", "EN_WAVG", "EN_LIN", "EN_LGBM", "EN_XGB", "EN_CAT"],
        "wavg_objective": "mse",
        "meta_trials": 15,
        "meta_val_fraction": 0.2,
        "meta_selection": "rolling",
        "meta_selection_windows": 4,
        "combiner_selection": "auto",
    },
    "evaluation": {
        "primary_metric": "rmse",
        "clip_min": 0.0,
        "band_quantiles": [0.25, 0.75],
        "band_labels": ["short", "medium", "long"],
        "period": "M",
        "rare_threshold": 10,
    },
    "statistics": {"alpha": 0.05, "bootstrap_resamples": 2000, "extra_references": []},
    "robustness": {"seeds": [0, 1, 2]},
    "rolling": {
        "enabled": True,
        "n_origins": 4,
        "test_fraction": 0.1,
        "window": "expanding",
        "tuning_mode": "single",
        "base_learners": "selected",
        "trial_scale": 0.5,
        "reference": "main",
    },
    "sensitivity": {
        "enabled": False,
        "learners": "selected",
        "params": None,
        "log_factors": [0.5, 2.0],
        "linear_step": 0.25,
    },
    "explain": {
        "enabled": True,
        "tree_models": "selected",
        "tree_shap_rows": 5000,
        "pipeline_shap_rows": 300,
        "pipeline_permutations": 3,
        "background_rows": 100,
        "top_k_overlap": 10,
        "dependence_top": 3,
        "waterfall_quantile": 0.95,
        "feature_labels": {},
        "value_labels": {},
    },
    "profiling": {
        "enabled": True,
        "latency_rows": 1000,
        "warmup": 50,
        "workers": [1, 2, 4, 8],
        "process_workers": [1, 2, 4, 8],
        "batch_rows": 100_000,
    },
}


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _unknown_keys(cfg: dict, ref: dict, prefix: str = "") -> list[str]:
    bad = []
    for key, value in cfg.items():
        if key not in ref:
            bad.append(prefix + key)
        elif isinstance(value, dict) and isinstance(ref[key], dict) and key not in {"n_trials", "cleaning", "calendar", "geo", "space_overrides", "feature_labels", "value_labels"}:
            bad += _unknown_keys(value, ref[key], prefix + key + ".")
    return bad


def _check_names(cfg: dict):
    """Reject unknown model names before any computation starts."""
    from .models import LEARNERS
    from .stacking import META_LEARNERS
    errors = []
    bases = cfg["base_learners"]
    errors += [f"base_learners: '{n}'" for n in bases if n not in LEARNERS]
    errors += [f"stacking.meta_learners: '{n}'" for n in cfg["stacking"]["meta_learners"] if n not in META_LEARNERS]
    errors += [f"tuning.n_trials: '{n}'" for n in cfg["tuning"]["n_trials"] if n not in LEARNERS]
    rb = cfg["rolling"]["base_learners"]
    if rb not in ("selected", "all"):
        errors += [f"rolling.base_learners: '{n}'" for n in rb if n not in LEARNERS]
    errors += [f"statistics.extra_references: '{n}'" for n in cfg["statistics"]["extra_references"]
               if n not in LEARNERS]
    ref = cfg["rolling"]["reference"]
    if ref not in ("main", "selected") and ref not in LEARNERS:
        errors += [f"rolling.reference: '{ref}'"]
    sl = cfg["sensitivity"]["learners"]
    if sl != "selected":
        errors += [f"sensitivity.learners: '{n}'" for n in sl if n not in LEARNERS]
    tm = cfg["explain"]["tree_models"]
    if tm != "selected":
        errors += [f"explain.tree_models: '{n}'" for n in tm if n not in ("LGBM", "XGB", "CAT")]
    if errors:
        raise ValueError("Unknown model names: " + "; ".join(errors)
                         + f". Base learners: {sorted(LEARNERS)}; combiners: {list(META_LEARNERS)}")
    if len(set(bases)) != len(bases):
        raise ValueError("base_learners contains duplicates")
    if cfg["run"]["stop_after"] is None and cfg["stacking"]["top_k"] > len(bases):
        raise ValueError("stacking.top_k exceeds the number of base learners")


def _read_user(path: Path, chain: tuple = ()) -> dict:
    """Read a configuration file; ``extends: <file>`` inherits every setting of another file."""
    path = Path(path).resolve()
    if path in chain:
        raise ValueError(f"Circular 'extends' chain: {' -> '.join(map(str, (*chain, path)))}")
    with open(path, "r", encoding="utf-8") as fh:
        user = yaml.safe_load(fh) or {}
    parent = user.pop("extends", None)
    if parent is None:
        return user
    return deep_merge(_read_user(path.parent / parent, (*chain, path)), user)


def load_config(path: str | Path, overrides: dict | None = None) -> dict:
    user = _read_user(Path(path))
    unknown = _unknown_keys(user, DEFAULTS)
    if unknown:
        raise KeyError(f"Unknown configuration keys: {unknown}")
    cfg = deep_merge(DEFAULTS, user)
    if overrides:
        cfg = deep_merge(cfg, overrides)
    if cfg["stacking"]["primary_scheme"] not in cfg["stacking"]["oof_schemes"]:
        raise ValueError("stacking.primary_scheme must be listed in stacking.oof_schemes")
    ev = cfg["evaluation"]
    if len(ev["band_labels"]) != len(ev["band_quantiles"]) + 1:
        raise ValueError("evaluation.band_labels needs exactly one label more than evaluation.band_quantiles")
    if cfg["split"]["n_folds"] < 2:
        raise ValueError("split.n_folds must be at least 2")
    d = cfg["dataset"]
    if d.get("test_target_path") and not d.get("test_path"):
        raise ValueError("dataset.test_target_path requires dataset.test_path")
    _check_names(cfg)
    if cfg["run"]["stop_after"] not in (None, "base_learners"):
        raise ValueError("run.stop_after must be null or 'base_learners'")
    if cfg["tuning"]["refit_scaling"] not in ("sqrt", "linear", "none"):
        raise ValueError("tuning.refit_scaling must be one of: sqrt, linear, none")
    if cfg["tuning"]["mode"] not in {"nested", "single", "none"}:
        raise ValueError("tuning.mode must be one of: nested, single, none")
    if d["target_transform"] not in TRANSFORMS:
        raise ValueError(f"dataset.target_transform must be one of: {', '.join(TRANSFORMS)}")
    if d["target_smearing"] not in SMEARING:
        raise ValueError(f"dataset.target_smearing must be one of: {', '.join(SMEARING)}")
    st = cfg["stacking"]
    if st["meta_selection"] not in {"rolling", "holdout"}:
        raise ValueError("stacking.meta_selection must be one of: rolling, holdout")
    if st["meta_selection"] == "rolling" and st["meta_selection_windows"] < 1:
        raise ValueError("stacking.meta_selection_windows must be at least 1")
    mv = cfg["tuning"]["max_val_rows"]
    if mv is not None and mv < 1:
        raise ValueError("tuning.max_val_rows must be null or a positive integer")
    if cfg["tuning"]["inner_split"] not in {"contiguous", "all"}:
        raise ValueError("tuning.inner_split must be one of: contiguous, all")
    if cfg["tuning"]["val_position"] not in {"latest", "after_cut"}:
        raise ValueError("tuning.val_position must be one of: latest, after_cut")
    if cfg["tuning"]["sampler_seed"] not in {"per_split", "shared"}:
        raise ValueError("tuning.sampler_seed must be one of: per_split, shared")
    from .models import resolve_space
    for key in cfg["tuning"]["space_overrides"] or {}:
        resolve_space(key, cfg["tuning"])
    if st["combiner_selection"] not in {"auto", "forward", "blocked"}:
        raise ValueError("stacking.combiner_selection must be one of: auto, forward, blocked")
    if st["combiner_selection"] == "auto":
        # forward chaining whenever it is computed: its meta-rows come from models that never saw a later row
        st["combiner_selection"] = "forward" if "forward" in st["oof_schemes"] else st["primary_scheme"]
    if st["combiner_selection"] not in st["oof_schemes"]:
        raise ValueError("stacking.combiner_selection must name a scheme listed in stacking.oof_schemes")
    return cfg


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:12]
