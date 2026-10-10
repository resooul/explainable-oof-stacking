"""Three-level SHAP analysis of the stacked ensemble.

1. Level-0: exact TreeSHAP of each gradient-boosting base learner on the original features.
2. Level-1: attribution of the meta-learner output to the base-learner predictions.
3. Composite: model-agnostic permutation SHAP of the full pipeline
   (raw features -> base learners -> meta-learner) on the original features.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy.stats import kendalltau, spearmanr

from .models import TabularEncoder
from .stacking import AverageMeta, LinearMeta, StackedPipeline, TreeMeta

def tree_shap(learner, X: pd.DataFrame) -> shap.Explanation:
    """Exact TreeSHAP values of a gradient-boosting base learner on the original features, computed by the model's
    own library.
    """
    values, base = learner.contributions(X)
    return shap.Explanation(values=values, base_values=np.broadcast_to(base, len(values)).copy(),
                            data=learner.encoder.to_numeric(X), display_data=_display(X, learner.features),
                            feature_names=list(learner.features))


def _display(X: pd.DataFrame, features) -> np.ndarray:
    """Original feature values of ``X`` as objects, used to label plots."""
    return X[list(features)].astype(object).to_numpy()


def meta_shap(meta, Z: np.ndarray, names: list[str], Z_reference: np.ndarray) -> shap.Explanation:
    """Exact interventional attributions for linear/averaging metas; TreeSHAP for tree metas."""
    Z = np.asarray(Z, float)
    if isinstance(meta, TreeMeta):
        values, base = meta.contributions(Z)
        base = np.broadcast_to(base, len(values)).copy()
    else:
        k = Z.shape[1]
        w = np.full(k, 1 / k) if isinstance(meta, AverageMeta) else np.asarray(meta.weights, float)
        mu = Z_reference.mean(axis=0)
        values = (Z - mu) * w
        intercept = meta.model.intercept_ if isinstance(meta, LinearMeta) else 0.0
        base = np.full(len(Z), float(mu @ w + intercept))
    return shap.Explanation(values=values, base_values=base, data=Z, feature_names=names)


def pipeline_shap(pipe: StackedPipeline, encoder: TabularEncoder, X_explain: pd.DataFrame,
                  X_background: pd.DataFrame, permutations: int, seed: int) -> shap.Explanation:
    """Permutation SHAP values of the complete pipeline, from the original features through the base learners and
    the combiner, on the model scale.

    Each row of ``X_explain`` is explained against the rows of ``X_background`` (interventional value function) with
    ``permutations`` antithetic permutations, i.e. ``permutations * (2 * F + 1)`` pipeline evaluations per row for F
    features.
    """
    A_bg, A = encoder.to_numeric(X_background), encoder.to_numeric(X_explain)
    # explained on the model scale, like the TreeSHAP explanations of the base learners it is compared with
    predict = getattr(pipe, "predict_model", pipe.predict)
    f = lambda M: predict(encoder.from_numeric(M))
    masker = shap.maskers.Independent(A_bg, max_samples=len(A_bg))
    explainer = shap.PermutationExplainer(f, masker, seed=seed)
    n_features = A.shape[1]
    ex = explainer(A, max_evals=max(permutations, 1) * (2 * n_features + 1), silent=True)
    return shap.Explanation(values=ex.values, base_values=ex.base_values, data=A,
                            display_data=_display(X_explain, encoder.features), feature_names=list(encoder.features))


def importance(ex: shap.Explanation) -> pd.Series:
    """Mean absolute SHAP value of every feature, in decreasing order."""
    return pd.Series(np.abs(ex.values).mean(axis=0), index=ex.feature_names).sort_values(ascending=False)


def agreement(a: pd.Series, b: pd.Series, k: int) -> dict:
    """Agreement of two importance rankings over their common features: Spearman's rho, Kendall's tau, the Jaccard
    overlap of the ``k`` most important features, and whether the most important feature is the same.
    """
    common = a.index.intersection(b.index)
    a, b = a[common], b[common]
    top_a, top_b = set(a.nlargest(k).index), set(b.nlargest(k).index)
    return {"spearman_rho": float(spearmanr(a, b).statistic), "kendall_tau": float(kendalltau(a, b).statistic),
            f"top{k}_jaccard": len(top_a & top_b) / len(top_a | top_b), "top1_match": a.idxmax() == b.idxmax()}


def _save(fig_path: str):
    """Save the current figure as PNG and PDF under ``fig_path`` and close it."""
    fig = plt.gcf()
    fig.savefig(fig_path + ".png", dpi=200, bbox_inches="tight")
    fig.savefig(fig_path + ".pdf", bbox_inches="tight")
    plt.close("all")


def _clean_axes(xlabel: str | None = None, ylabel: str | None = None):
    """Remove titles from every axis of the current figure and set plain axis labels."""
    fig = plt.gcf()
    fig.suptitle("")
    for ax in fig.axes:
        ax.set_title("")
    ax = fig.axes[0] if fig.axes else plt.gca()
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)


def relabel(ex: shap.Explanation, labels: dict | None, value_labels: dict | None = None) -> shap.Explanation:
    """Copy of ``ex`` with display labels: ``labels`` renames features, ``value_labels`` renames the values of
    categorical features as shown in waterfall and dependence plots, e.g. {shipmentStatus: {delivered: Delivery}}.
    Values are matched by their text, so YAML keys need no quoting; unlisted names and values are kept."""
    if not labels and not value_labels:
        return ex
    labels, value_labels = labels or {}, value_labels or {}
    display = getattr(ex, "display_data", None)
    if display is not None and value_labels:
        display = np.array(display, dtype=object, copy=True)
        for j, f in enumerate(ex.feature_names):
            vmap = {str(k): v for k, v in (value_labels.get(f) or {}).items()}
            if vmap:
                display[:, j] = [vmap.get(str(v), v) for v in display[:, j]]
    return shap.Explanation(values=ex.values, base_values=ex.base_values, data=ex.data, display_data=display,
                            feature_names=[labels.get(f, f) for f in ex.feature_names])


def save_explanation(path, ex: shap.Explanation, **plot_args):
    """Store an explanation with its plot settings so that its figures can be redrawn without recomputation."""
    import joblib
    joblib.dump({"explanation": ex, "plot_args": plot_args}, path)


def plot_explanation(ex: shap.Explanation, prefix: str, dependence_top: int = 0, waterfall_index: int | None = None,
                     max_display: int = 20, labels: dict | None = None, value_labels: dict | None = None):
    """Bar and beeswarm plots of ``ex``, dependence plots of its ``dependence_top`` most important features and,
    with ``waterfall_index``, the waterfall plot of that row.

    Files are named ``<prefix>_<plot>``; dependence plots keep the column name of the feature in their file name
    even when ``labels`` gives it a display name.
    """
    # figures carry no titles and plain axis labels, so that captions can describe them
    order = list(importance(ex).index[:dependence_top])          # raw names, used for file names
    raw_names = list(ex.feature_names)
    ex = relabel(ex, labels, value_labels)
    plt.figure()
    shap.plots.bar(ex, max_display=max_display, show=False)
    _clean_axes("Mean |SHAP value|")
    _save(prefix + "_bar")
    plt.figure()
    shap.plots.beeswarm(ex, max_display=max_display, show=False)
    _clean_axes("SHAP value")
    _save(prefix + "_beeswarm")
    names = list(ex.feature_names)
    for raw in order:
        j = raw_names.index(raw)
        plt.figure()
        shap.plots.scatter(ex[:, j], color=ex, show=False)
        _clean_axes(None, f"SHAP value for {names[j]}")
        _save(f"{prefix}_dependence_{raw}")
    if waterfall_index is not None:
        plt.figure()
        shap.plots.waterfall(ex[waterfall_index], max_display=15, show=False)
        _save(prefix + "_waterfall")


def seed_stability(importances: dict[int, pd.Series]) -> pd.DataFrame:
    """Pairwise agreement of the importance rankings obtained with different random seeds."""
    seeds = sorted(importances)
    rows = []
    for i, a in enumerate(seeds):
        for b in seeds[i + 1:]:
            rows.append({"seed_a": a, "seed_b": b, **agreement(importances[a], importances[b], 10)})
    return pd.DataFrame(rows)
