import importlib.util

import yaml

from stackxai.config import load_config
from stackxai.pipeline import run
from synthetic import make_nyc_like

DEEP = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("rtdl_revisiting_models") is not None


def test_end_to_end(tmp_path):
    csv = tmp_path / "train.csv"
    make_nyc_like(4000).to_csv(csv, index=False)
    user = {
        "run": {"name": "smoke", "output_dir": str(tmp_path / "results"), "n_jobs": 1},
        "dataset": {"adapter": "nyc_taxi", "path": str(csv)},
        "split": {"n_folds": 3},
        "tuning": {"max_rows": 1500, "early_stopping_rounds": 10, "max_estimators": 100,
                   "n_trials": {"LGBM": 2, "XGB": 2, "CAT": 1, "DT": 1, "RIDGE": 1, "MLP": 1},
                   "deep": {"max_epochs": 3, "patience": 1, "batch_size": 256}},
        "base_learners": ["LGBM", "XGB", "CAT", "DT", "RIDGE", *(["MLP"] if DEEP else [])],
        "stacking": {"meta_trials": 2, "meta_learners": ["EN_AVG", "EN_WAVG", "EN_LIN", "EN_SLR", "EN_LGBM",
                                                         "EN_XGB", "EN_CAT"]},
        "statistics": {"bootstrap_resamples": 50},
        "robustness": {"seeds": [0, 1]},
        "rolling": {"n_origins": 2},
        "explain": {"tree_shap_rows": 200, "pipeline_shap_rows": 5, "pipeline_permutations": 1,
                    "background_rows": 10},
        "sensitivity": {"enabled": True, "params": ["learning_rate"]},
        "profiling": {"latency_rows": 20, "warmup": 2, "workers": [1], "process_workers": [1], "batch_rows": 500},
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(user))
    run_dir = run(load_config(path), resume=False)
    for name in ("test_performance", "statistical_tests", "shap_agreement", "seed_robustness",
                 "rolling_origin_summary", "rolling_selection", "rolling_comparison", "oof_scheme_comparison",
                 "oof_by_fold", "performance_by_category_frequency", "hyperparameter_sensitivity", "profiling_latency",
                 "profiling_meta", "combiner_selection", "combiner_selection_windows", "shap_seed_stability",
                 "categorical_prevalence", "categorical_levels"):
        assert (run_dir / "tables" / f"{name}.csv").exists(), name
    assert (run_dir / "SUMMARY.md").exists()
