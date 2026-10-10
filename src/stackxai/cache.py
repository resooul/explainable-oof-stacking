"""Content-addressed checkpointing of expensive computation units."""
from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

import joblib
import numpy as np

log = logging.getLogger(__name__)


def fingerprint(*parts) -> str:
    def default(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.integer, np.floating)):
            return o.item()
        return str(o)
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=default).encode()).hexdigest()[:16]


class Cache:
    """Stores each unit under ``<root>/<key>-<fingerprint>.joblib``.

    The fingerprint covers every input that can change the unit's result, so a
    changed setting recomputes only the affected units. Files are written
    atomically, hence an interrupted run never leaves a corrupt checkpoint.
    """

    def __init__(self, root: Path | None, enabled: bool = True):
        self.root, self.enabled = (Path(root) if root else None), enabled
        self.hits = self.misses = 0

    def get(self, key: str, fn, fp: str = ""):
        if self.root is None:
            return fn()
        path = self.root / f"{key}-{fp}.joblib"
        if self.enabled and path.exists():
            try:
                obj = joblib.load(path)
                self.hits += 1
                return obj
            except Exception as exc:
                log.warning("cache | unreadable checkpoint %s (%s); recomputing", path.name, exc)
        obj = fn()
        self.misses += 1
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        try:
            joblib.dump(obj, tmp)
            os.replace(tmp, path)
        except Exception as exc:
            log.warning("cache | could not store %s (%s)", path.name, exc)
            tmp.unlink(missing_ok=True)
        return obj


# Settings introduced after checkpoints were first written. At their neutral value they do not change
# any result, so they are left out of fingerprints and existing checkpoints stay valid.
NEUTRAL_SETTINGS = {"tuning": {"max_val_rows": None, "val_position": "after_cut", "sampler_seed": "shared",
                               "space_overrides": {}},
                    "dataset": {"drop_features": [], "target_transform": "none"}}
# Settings that only affect how results are reported, never what is computed and checkpointed.
REPORT_ONLY = {"dataset": {"target_smearing"}}


def stable_section(cfg: dict, section: str) -> dict:
    """``cfg[section]`` without settings that are at their neutral (result-preserving) value."""
    neutral = NEUTRAL_SETTINGS.get(section, {})
    skip = REPORT_ONLY.get(section, set())
    return {k: v for k, v in cfg[section].items() if not (k in neutral and v == neutral[k]) and k not in skip}
