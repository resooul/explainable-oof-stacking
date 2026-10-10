"""Base and meta learners with a uniform, leakage-safe interface.

Each learner owns its preprocessing, which is fitted exclusively on the data
passed to ``fit``. Search spaces are declarative so that they can be both
sampled by Optuna and exported as a reproducibility table.
"""
from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd

SPACES: dict[str, dict] = {
    "LGBM": {"num_leaves": ("int_log", 15, 1023), "learning_rate": ("float_log", 0.02, 0.2),
             "min_child_samples": ("int_log", 5, 1000), "subsample": ("float", 0.6, 1.0),
             "colsample_bytree": ("float", 0.4, 1.0), "reg_lambda": ("float_log", 1e-3, 10.0)},
    "XGB": {"max_depth": ("int", 4, 18), "learning_rate": ("float_log", 0.02, 0.2),
            "min_child_weight": ("float_log", 1.0, 1000.0), "subsample": ("float", 0.6, 1.0),
            "colsample_bytree": ("float", 0.3, 1.0), "reg_lambda": ("float_log", 1e-3, 10.0)},
    "CAT": {"depth": ("int", 4, 12), "learning_rate": ("float_log", 0.03, 0.3),
            "l2_leaf_reg": ("float_log", 1.0, 30.0), "random_strength": ("float_log", 0.01, 10.0)},
    "RF": {"max_depth": ("int", 8, 32), "min_samples_leaf": ("int_log", 1, 100),
           "max_features": ("float", 0.3, 1.0)},
    "DT": {"max_depth": ("int", 4, 32), "min_samples_leaf": ("int_log", 1, 500)},
    "RIDGE": {"alpha": ("float_log", 1e-3, 1e3)},
    "EBM": {"interactions": ("int", 0, 30), "learning_rate": ("float_log", 0.005, 0.1),
            "min_samples_leaf": ("int_log", 2, 100), "max_bins": ("int_log", 64, 1024), "max_leaves": ("int", 2, 4)},
    "MLP": {"n_blocks": ("int", 1, 6), "d_block": ("int_log", 64, 512), "dropout": ("float", 0.0, 0.5),
            "learning_rate": ("float_log", 1e-4, 3e-3), "weight_decay": ("float_log", 1e-6, 1e-3)},
    "RESNET": {"n_blocks": ("int", 1, 4), "d_block": ("int_log", 64, 512), "dropout": ("float", 0.0, 0.5),
               "learning_rate": ("float_log", 1e-4, 3e-3), "weight_decay": ("float_log", 1e-6, 1e-3)},
    "CNN": {"dropout": ("float", 0.0, 0.4), "learning_rate": ("float_log", 1e-4, 3e-3),
            "weight_decay": ("float_log", 1e-6, 1e-3)},
    "LSTM": {"dropout": ("float", 0.0, 0.4), "learning_rate": ("float_log", 1e-4, 3e-3),
             "weight_decay": ("float_log", 1e-6, 1e-3)},
    "FTT": {"n_blocks": ("int", 1, 4), "d_block": ("int", 64, 256, 8), "attention_dropout": ("float", 0.0, 0.5),
            "ffn_dropout": ("float", 0.0, 0.5), "learning_rate": ("float_log", 1e-6, 1e-3),
            "weight_decay": ("float_log", 1e-6, 1e-3)},
    "TABNET": {"n_d": ("int", 8, 64), "n_steps": ("int", 2, 10), "gamma": ("float", 1.0, 2.0),
               "lambda_sparse": ("float_log", 1e-6, 1e-3), "learning_rate": ("float_log", 5e-3, 5e-2)},
    "META_LGBM": {"num_leaves": ("int", 4, 63), "learning_rate": ("float_log", 0.02, 0.2),
                  "min_child_samples": ("int_log", 20, 1000)},
    "META_XGB": {"max_depth": ("int", 2, 8), "learning_rate": ("float_log", 0.02, 0.2),
                 "min_child_weight": ("float_log", 1.0, 100.0)},
    "META_CAT": {"depth": ("int", 2, 8), "learning_rate": ("float_log", 0.03, 0.3),
                 "l2_leaf_reg": ("float_log", 1.0, 30.0)},
}


def sample_space(trial, space: dict) -> dict:
    params = {}
    for name, (kind, lo, hi, *step) in space.items():
        if kind == "int":
            params[name] = trial.suggest_int(name, lo, hi, step=step[0] if step else 1)
        elif kind == "int_log":
            params[name] = trial.suggest_int(name, lo, hi, log=True)
        elif kind == "float":
            params[name] = trial.suggest_float(name, lo, hi)
        elif kind == "float_log":
            params[name] = trial.suggest_float(name, lo, hi, log=True)
        else:
            raise ValueError(kind)
    return params


def resolve_space(key: str, tuning: dict | None = None) -> dict:
    """Search space of ``key`` with the overrides of ``tuning.space_overrides`` applied.

    ``space_overrides: {RIDGE: {alpha: [float_log, 0.001, 1.0e7]}}`` replaces or adds single parameters; the
    remaining parameters keep their built-in ranges."""
    space = dict(SPACES.get(key, {}))
    for param, spec in ((tuning or {}).get("space_overrides") or {}).get(key, {}).items():
        spec = list(spec)
        try:
            kind = spec[0]
            cast = int if kind.startswith("int") else float     # YAML reads e.g. 1.0e7 (no exponent sign) as text
            lo, hi = cast(float(spec[1])), cast(float(spec[2]))
            ok = kind in ("int", "int_log", "float", "float_log") and lo < hi and (lo > 0 or "log" not in kind)
        except (IndexError, TypeError, ValueError):
            ok = False
        if not ok:
            raise ValueError(f"tuning.space_overrides.{key}.{param}: expected [kind, low, high], got {spec}")
        space[param] = (kind, lo, hi, *spec[3:])
    return space


def space_table(tuning: dict | None = None) -> pd.DataFrame:
    keys = list(SPACES) + [k for k in ((tuning or {}).get("space_overrides") or {}) if k not in SPACES]
    rows = [{"model": m, "parameter": p, "scale": k.replace("_log", " (log)"), "low": lo, "high": hi,
             "step": s[0] if s else None}
            for m in keys for p, (k, lo, hi, *s) in resolve_space(m, tuning).items()]
    return pd.DataFrame(rows)


class TabularEncoder:
    """Fits categorical vocabularies on training data only; unseen levels become missing."""

    def __init__(self, features: list[str], categorical: list[str]):
        self.features = list(features)
        self.categorical = [c for c in categorical if c in self.features]
        self.numeric = [c for c in self.features if c not in self.categorical]
        self.categories: dict[str, pd.Index] = {}

    @staticmethod
    def _plain(s: pd.Series) -> pd.Series:
        # a pandas 'category' column (e.g. read from parquet) keeps unused levels in its dtype; working on the
        # plain values makes the vocabulary consist of the observed levels only, so that codes always lie in
        # 0..len(vocabulary)-1 (embedding layers index by these codes)
        if isinstance(s.dtype, pd.CategoricalDtype):
            dtype = s.cat.categories.dtype
            if dtype != "category":
                try:
                    return s.astype(dtype)           # the conversion used when the models were trained
                except (TypeError, ValueError):      # integer categories with missing values (e.g. rows rebuilt
                    pass                             # from codes for permutation SHAP): keep values as objects
            return pd.Series(np.asarray(s, dtype=object), index=s.index, name=s.name)
        return s

    def fit(self, X: pd.DataFrame) -> "TabularEncoder":
        self.categories = {c: pd.Index(pd.unique(self._plain(X[c]).dropna())).sort_values()
                           for c in self.categorical}
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X[self.features].copy()
        for c in self.numeric:
            out[c] = out[c].astype(float)
        for c in self.categorical:
            cats = self.categories[c]
            if isinstance(cats, pd.CategoricalIndex):          # vocabulary pickled by an earlier version
                cats = pd.Index(np.asarray(cats))
            v = self._plain(out[c])
            out[c] = pd.Categorical(v.where(v.isin(cats)), categories=cats)
        return out

    def same_vocabulary(self, other: "TabularEncoder") -> bool:
        return (self.features == other.features and self.categorical == other.categorical
                and all(self.categories[c].equals(other.categories[c]) for c in self.categorical))

    def to_numeric(self, X: pd.DataFrame) -> np.ndarray:
        """Float matrix with categorical codes (missing/unseen -> NaN)."""
        return self.encoded_to_numeric(self.transform(X))

    def encoded_to_numeric(self, Z: pd.DataFrame) -> np.ndarray:
        """``to_numeric`` for rows that have already been transformed."""
        A = np.empty((len(Z), len(self.features)), dtype=float)
        for j, c in enumerate(self.features):
            if c in self.categories:
                codes = Z[c].cat.codes.to_numpy().astype(float)
                codes[codes < 0] = np.nan
                A[:, j] = codes
            else:
                A[:, j] = Z[c].to_numpy(dtype=float)
        return A

    def from_numeric(self, A: np.ndarray) -> pd.DataFrame:
        cols = {}
        for j, c in enumerate(self.features):
            if c in self.categories:
                codes = np.nan_to_num(A[:, j], nan=-1).round().astype(int)
                cols[c] = pd.Categorical.from_codes(np.clip(codes, -1, len(self.categories[c]) - 1),
                                                    categories=self.categories[c])
            else:
                cols[c] = A[:, j]
        return pd.DataFrame(cols)


def as_labels(col: pd.Series) -> pd.Series:
    """String labels with an explicit token for missing or unseen levels."""
    return col.astype(object).where(col.notna(), "__missing__").astype(str)


class Learner:
    name = "base"
    space_key: str | None = None
    iterative = False
    shares_encoding = False   # predict() is encoder.transform() followed by _predict(): the encoding can be shared

    def __init__(self, features, categorical, params=None, seed=0, n_jobs=-1, cfg=None):
        self.features, self.categorical = list(features), list(categorical)
        self.params = dict(params or {})
        self.seed, self.n_jobs = seed, n_jobs
        self.cfg = cfg or {}
        self.encoder = TabularEncoder(self.features, self.categorical)
        self.model = None
        self.best_iteration: int | None = None

    @property
    def space(self) -> dict:
        return resolve_space(self.space_key or self.name, self.cfg)

    def fit(self, X, y, X_val=None, y_val=None) -> "Learner":
        self.encoder.fit(X)
        self._fit(self.encoder.transform(X), np.asarray(y, float),
                  None if X_val is None else self.encoder.transform(X_val),
                  None if y_val is None else np.asarray(y_val, float))
        return self

    def predict(self, X) -> np.ndarray:
        return np.asarray(self._predict(self.encoder.transform(X)), dtype=float).ravel()

    def predict_encoded(self, Z: pd.DataFrame) -> np.ndarray:
        """Prediction from rows already passed through an encoder identical to ``self.encoder``."""
        return np.asarray(self._predict(Z), dtype=float).ravel()

    def set_inference_threads(self, n: int):
        pass

    def _fit(self, X, y, X_val, y_val):
        raise NotImplementedError

    def _predict(self, X):
        return self.model.predict(X)


def _n_estimators(learner: Learner, has_val: bool) -> int:
    if has_val:
        return int(learner.cfg.get("max_estimators", 5000))
    return int(learner.params.get("n_estimators", 1000))


def _lgbm_supports_eval_xy() -> bool:
    try:
        import inspect

        import lightgbm as lgb
        return "eval_X" in inspect.signature(lgb.LGBMRegressor.fit).parameters
    except ImportError:
        return False


_LGBM_EVAL_XY = _lgbm_supports_eval_xy()


class LGBMLearner(Learner):
    name, iterative, shares_encoding = "LGBM", True, True

    def _fit(self, X, y, X_val, y_val):
        import lightgbm as lgb
        p = {k: v for k, v in self.params.items() if k != "n_estimators"}
        self.model = lgb.LGBMRegressor(n_estimators=_n_estimators(self, X_val is not None), subsample_freq=1,
                                       random_state=self.seed, n_jobs=self.n_jobs, verbose=-1, **p)
        kw = {}
        if X_val is not None:
            kw = {"eval_X": (X_val,), "eval_y": (y_val,)} if _LGBM_EVAL_XY else {"eval_set": [(X_val, y_val)]}
            kw["callbacks"] = [lgb.early_stopping(self.cfg.get("early_stopping_rounds", 100), verbose=False)]
        self.model.fit(X, y, **kw)
        self.best_iteration = int(self.model.best_iteration_ or self.model.n_estimators)

    def _predict(self, X):
        return self.model.predict(X, num_iteration=self.best_iteration)

    def predict_encoded(self, Z):
        # the booster maps pandas categories to the codes of its training frame; the encoder's vocabulary is
        # that frame's vocabulary, so its codes (unseen -> NaN) can be passed directly, avoiding pandas overhead
        A = self.encoder.encoded_to_numeric(Z)
        return np.asarray(self.model.booster_.predict(A, num_iteration=self.best_iteration,
                                                      num_threads=getattr(self, "_threads", 0)), dtype=float).ravel()

    def set_inference_threads(self, n):
        self.model.set_params(n_jobs=n)
        self._threads = n if n and n > 0 else 0

    def contributions(self, X):
        c = self.model.predict(self.encoder.transform(X), pred_contrib=True, num_iteration=self.best_iteration)
        return c[:, :-1], c[:, -1]


class XGBLearner(Learner):
    name, iterative, shares_encoding = "XGB", True, True

    def _fit(self, X, y, X_val, y_val):
        import xgboost as xgb
        p = {k: v for k, v in self.params.items() if k != "n_estimators"}
        es = self.cfg.get("early_stopping_rounds", 100) if X_val is not None else None
        self.model = xgb.XGBRegressor(n_estimators=_n_estimators(self, X_val is not None), tree_method="hist",
                                      enable_categorical=True, max_cat_to_onehot=1, early_stopping_rounds=es,
                                      random_state=self.seed, n_jobs=self.n_jobs, **p)
        kw = {"eval_set": [(X_val, y_val)], "verbose": False} if X_val is not None else {}
        self.model.fit(X, y, **kw)
        self.best_iteration = int(self.model.best_iteration + 1) if es else int(self.model.n_estimators)

    def _predict(self, X):
        return self.model.predict(X, iteration_range=(0, self.best_iteration))

    def set_inference_threads(self, n):
        self.model.set_params(n_jobs=n)

    def contributions(self, X):
        import xgboost as xgb
        d = xgb.DMatrix(self.encoder.transform(X), enable_categorical=True)
        c = self.model.get_booster().predict(d, pred_contribs=True, iteration_range=(0, self.best_iteration))
        return c[:, :-1], c[:, -1]


class CATLearner(Learner):
    name, iterative, shares_encoding = "CAT", True, True

    def _prep(self, X):
        X = X.copy()
        for c in self.encoder.categorical:
            X[c] = as_labels(X[c])
        return X

    def _fit(self, X, y, X_val, y_val):
        from catboost import CatBoostRegressor
        p = {k: v for k, v in self.params.items() if k != "n_estimators"}
        threads = self.n_jobs if self.n_jobs and self.n_jobs > 0 else -1
        self.model = CatBoostRegressor(iterations=_n_estimators(self, X_val is not None), loss_function="RMSE",
                                       random_seed=self.seed, thread_count=threads, verbose=0,
                                       allow_writing_files=False, cat_features=self.encoder.categorical, **p)
        kw = {}
        if X_val is not None:
            kw = {"eval_set": (self._prep(X_val), y_val),
                  "early_stopping_rounds": self.cfg.get("early_stopping_rounds", 100)}
        self.model.fit(self._prep(X), y, **kw)
        self.best_iteration = int(self.model.get_best_iteration() + 1) if X_val is not None else int(self.model.tree_count_)

    def _predict(self, X):
        return self.model.predict(self._prep(X), thread_count=getattr(self, "_threads", -1))

    def set_inference_threads(self, n):
        self._threads = n

    def contributions(self, X):
        from catboost import Pool
        pool = Pool(self._prep(self.encoder.transform(X)), cat_features=self.encoder.categorical)
        c = self.model.get_feature_importance(pool, type="ShapValues")
        return c[:, :-1], c[:, -1]


class _SklearnTree(Learner):
    def _array(self, X):
        return self.encoder.to_numeric(X)

    def fit(self, X, y, X_val=None, y_val=None):
        self.encoder.fit(X)
        self.model = self._build()
        self.model.fit(self._array(X), np.asarray(y, float))
        return self

    def predict(self, X):
        return self.model.predict(self._array(X))

    def set_inference_threads(self, n):
        if hasattr(self.model, "n_jobs"):
            self.model.set_params(n_jobs=n)


class RFLearner(_SklearnTree):
    name = "RF"

    def _build(self):
        from sklearn.ensemble import RandomForestRegressor
        p = {"n_estimators": 300, "max_samples": 0.5, **self.params}
        return RandomForestRegressor(random_state=self.seed, n_jobs=self.n_jobs, **p)


class DTLearner(_SklearnTree):
    name = "DT"

    def _build(self):
        from sklearn.tree import DecisionTreeRegressor
        return DecisionTreeRegressor(random_state=self.seed, **self.params)


class _Linear(Learner):
    def _pipeline(self, estimator):
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        num = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler())
        cat = OneHotEncoder(handle_unknown="infrequent_if_exist", max_categories=50, sparse_output=False)
        ct = ColumnTransformer([("num", num, self.encoder.numeric), ("cat", cat, self.encoder.categorical)])
        return make_pipeline(ct, estimator)

    def _frame(self, X):
        X = self.encoder.transform(X)
        for c in self.encoder.categorical:
            X[c] = as_labels(X[c])
        return X

    def fit(self, X, y, X_val=None, y_val=None):
        self.encoder.fit(X)
        self.model = self._pipeline(self._estimator())
        self.model.fit(self._frame(X), np.asarray(y, float))
        return self

    def predict(self, X):
        return self.model.predict(self._frame(X))


class OLSLearner(_Linear):
    name = "OLS"

    def _estimator(self):
        from sklearn.linear_model import LinearRegression
        return LinearRegression()


class RidgeLearner(_Linear):
    name = "RIDGE"

    def _estimator(self):
        from sklearn.linear_model import Ridge
        return Ridge(alpha=self.params.get("alpha", 1.0))


class EBMLearner(Learner):
    """Explainable Boosting Machine (a GA2M), an interpretable-by-design baseline."""
    name = "EBM"

    def fit(self, X, y, X_val=None, y_val=None):
        try:
            from interpret.glassbox import ExplainableBoostingRegressor
        except ImportError as exc:
            raise ImportError("EBM requires the optional dependency 'interpret'") from exc
        self.encoder.fit(X)
        self.model = ExplainableBoostingRegressor(random_state=self.seed, n_jobs=self.n_jobs,
                                                  **({"interactions": 10} | self.params))
        self.model.fit(self.encoder.transform(X), np.asarray(y, float))
        return self

    def predict(self, X):
        return self.model.predict(self.encoder.transform(X))


from .deep import (CNNLearner, FTTransformerLearner, LSTMLearner, MLPLearner,  # noqa: E402
                   ResNetLearner, TabNetLearner)

LEARNERS: dict[str, type[Learner]] = {cls.name: cls for cls in
                                      (LGBMLearner, XGBLearner, CATLearner, RFLearner, DTLearner, OLSLearner,
                                       RidgeLearner, EBMLearner, MLPLearner, ResNetLearner, CNNLearner,
                                       LSTMLearner, FTTransformerLearner, TabNetLearner)}
DEEP_LEARNERS = {"MLP", "RESNET", "CNN", "LSTM", "FTT", "TABNET"}


def make_learner(name: str, features, categorical, params=None, seed=0, n_jobs=-1, cfg=None) -> Learner:
    if name not in LEARNERS:
        raise KeyError(f"Unknown learner '{name}'. Available: {sorted(LEARNERS)}")
    return LEARNERS[name](features, categorical, params, seed, n_jobs, cfg)


REFIT_SCALINGS = ("sqrt", "linear", "none")


def refit_params(name: str, params: dict, best_iteration: int | None, n_inner: int, n_full: int,
                 tuning: dict | None = None) -> dict:
    """Boosting rounds for a refit on ``n_full`` rows, given the early-stopped count on ``n_inner`` rows.

    The count grows with the square root of the sample-size ratio (``tuning.refit_scaling``; ``linear`` and
    ``none`` are alternatives) and is capped at ``tuning.max_estimators``. Epochs of deep learners are not scaled.
    """
    tuning = tuning or {}
    out = dict(params)
    if best_iteration:
        ratio = max(1.0, n_full / max(n_inner, 1))
        mode = tuning.get("refit_scaling", "sqrt")
        if name in DEEP_LEARNERS or mode == "none":
            factor = 1.0
        elif mode == "linear":
            factor = ratio
        elif mode == "sqrt":
            factor = math.sqrt(ratio)
        else:
            raise ValueError(f"tuning.refit_scaling must be one of {REFIT_SCALINGS}")
        rounds = int(math.ceil(best_iteration * factor))
        cap = tuning.get("max_estimators")
        if cap and name not in DEEP_LEARNERS:
            rounds = min(rounds, int(cap))
        out["n_estimators"] = max(1, rounds)
    return out


warnings.filterwarnings("ignore", message=".*X does not have valid feature names.*")
