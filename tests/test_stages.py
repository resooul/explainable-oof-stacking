import importlib.util

import pytest
import yaml

from stackxai.config import load_config
from stackxai.pipeline import run
from synthetic import make_nyc_like

DEEP = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("rtdl_revisiting_models") is not None


def _write(path, content):
    path.write_text(yaml.safe_dump(content) if isinstance(content, dict) else content)
    return path


def test_staged_runs_reuse_every_base_learner(tmp_path):
    csv = tmp_path / "train.csv"
    make_nyc_like(3000).to_csv(csv, index=False)
    second = "MLP" if DEEP else "OLS"
    base = {
        "run": {"name": "staged", "output_dir": str(tmp_path / "results"), "n_jobs": 1},
        "dataset": {"adapter": "nyc_taxi", "path": str(csv)},
        "split": {"n_folds": 3},
        "tuning": {"max_rows": 1000, "early_stopping_rounds": 5, "max_estimators": 50,
                   "n_trials": {"DT": 2, "RIDGE": 2, "MLP": 1}, "deep": {"max_epochs": 2, "patience": 1}},
        "base_learners": ["DT", "RIDGE", second],
        "stacking": {"top_k": 2, "oof_schemes": ["blocked"], "meta_learners": ["EN_AVG", "EN_LIN"]},
        "robustness": {"seeds": [0]}, "rolling": {"enabled": False},
        "explain": {"enabled": False}, "profiling": {"enabled": False},
    }
    full = _write(tmp_path / "full.yaml", base)
    stage_a = _write(tmp_path / "a.yaml", "extends: full.yaml\nrun: {stop_after: base_learners}\nbase_learners: [DT, RIDGE]\n")
    stage_b = _write(tmp_path / "b.yaml", f"extends: full.yaml\nrun: {{stop_after: base_learners}}\nbase_learners: [{second}]\n")

    run(load_config(stage_a))
    run(load_config(stage_b))
    run_dir = run(load_config(full))
    summary = (run_dir / "SUMMARY.md").read_text()
    assert "15 checkpointed units restored, 0 computed" in summary
    assert (run_dir / "tables" / "test_performance.csv").exists()
    assert (run_dir / "tables" / "stage_dt_ridge.csv").exists()


def test_circular_extends_is_rejected(tmp_path):
    _write(tmp_path / "a.yaml", "extends: b.yaml\n")
    _write(tmp_path / "b.yaml", "extends: a.yaml\n")
    with pytest.raises(ValueError, match="Circular"):
        load_config(tmp_path / "a.yaml")
