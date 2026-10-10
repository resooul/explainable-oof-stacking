"""Paired statistical comparison of forecast errors."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def holm(pvalues: np.ndarray) -> np.ndarray:
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def cluster_bootstrap(y, p_ref, p_alt, clusters, n_resamples=2000, seed=0) -> dict:
    """Percentile CIs of MAE/RMSE differences (reference minus alternative), resampling whole clusters."""
    codes, _ = pd.factorize(clusters if clusters is not None else np.arange(len(y)))
    k = codes.max() + 1
    cnt = np.bincount(codes, minlength=k).astype(float)
    a_ref = np.bincount(codes, np.abs(y - p_ref), k)
    a_alt = np.bincount(codes, np.abs(y - p_alt), k)
    s_ref = np.bincount(codes, (y - p_ref) ** 2, k)
    s_alt = np.bincount(codes, (y - p_alt) ** 2, k)
    rng = np.random.default_rng(seed)
    d_mae, d_rmse = np.empty(n_resamples), np.empty(n_resamples)
    for b in range(n_resamples):
        w = np.bincount(rng.integers(0, k, k), minlength=k).astype(float)
        n = w @ cnt
        d_mae[b] = (w @ a_ref - w @ a_alt) / n
        d_rmse[b] = np.sqrt(w @ s_ref / n) - np.sqrt(w @ s_alt / n)
    point_mae = np.mean(np.abs(y - p_ref)) - np.mean(np.abs(y - p_alt))
    point_rmse = np.sqrt(np.mean((y - p_ref) ** 2)) - np.sqrt(np.mean((y - p_alt) ** 2))
    return {"dMAE": point_mae, "dMAE_lo": np.percentile(d_mae, 2.5), "dMAE_hi": np.percentile(d_mae, 97.5),
            "dRMSE": point_rmse, "dRMSE_lo": np.percentile(d_rmse, 2.5), "dRMSE_hi": np.percentile(d_rmse, 97.5),
            "n_clusters": int(k)}


def paired_comparison(y, preds: dict[str, np.ndarray], reference: str, clusters=None, strata=None,
                      alpha=0.05, n_resamples=2000, seed=0) -> pd.DataFrame:
    """Wilcoxon signed-rank test on absolute errors, cluster-bootstrap CIs and paired Cohen's d_z,
    with Holm correction across all comparisons against the reference model."""
    y = np.asarray(y, float)
    groups = [("all", np.ones(len(y), bool))]
    if strata is not None:
        groups += [(str(s), np.asarray(strata) == s) for s in pd.unique(strata)]
    rows = []
    for label, m in groups:
        ref_err = np.abs(y[m] - preds[reference][m])
        for name, p in preds.items():
            if name == reference:
                continue
            alt_err = np.abs(y[m] - p[m])
            d = ref_err - alt_err
            if np.all(d == 0):
                stat, pval = 0.0, 1.0
            else:
                stat, pval = wilcoxon(ref_err, alt_err, zero_method="wilcox", method="approx")
            boot = cluster_bootstrap(y[m], preds[reference][m], p[m],
                                     None if clusters is None else np.asarray(clusters)[m], n_resamples, seed)
            rows.append({"stratum": label, "reference": reference, "model": name, "n": int(m.sum()),
                         "wilcoxon_W": float(stat), "p_value": float(pval),
                         "cohens_dz": float(d.mean() / d.std(ddof=1)) if d.std() > 0 else 0.0, **boot})
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = holm(out["p_value"].to_numpy())
        out["significant"] = out["p_holm"] < alpha
        for m in ("MAE", "RMSE"):
            lo, hi = out[f"d{m}_lo"], out[f"d{m}_hi"]
            out[f"{m}_ci_excludes_zero"] = (lo > 0) | (hi < 0)
            # differences are reference minus model: positive means the model has the lower error
            out[f"{m}_direction"] = np.select([lo > 0, hi < 0], ["lower error", "higher error"], "inconclusive")
    return out
