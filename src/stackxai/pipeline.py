"""End-to-end experiment orchestration."""
from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from . import explain, metrics, reporting
from .cache import Cache
from .config import config_hash
from .data import Dataset, load_dataset
from .models import TabularEncoder, space_table
from .splits import describe_folds, given_partitions, holdout, rolling_origins
from .stacking import StackResult, StackedPipeline, frozen_params, run_stack
from .stats import paired_comparison
from .target import smearing_factors, to_report

log = logging.getLogger("stackxai")
# Packages whose installed versions are recorded in environment.json.
PACKAGES = ["numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "xgboost", "catboost", "optuna", "shap",
            "torch", "pytorch-tabnet", "rtdl_revisiting_models", "interpret"]


def _setup(cfg: dict) -> Path:
    """Create the output directories of the run, log to the console and to ``run.log``, and record the resolved
    configuration, the installed package versions, the configuration hash and the git commit.
    """
    run_dir = Path(cfg["run"]["output_dir"]) / cfg["run"]["name"]
    for sub in ("tables", "figures", "models", "predictions"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in list(root.handlers):
        h.close()
        root.removeHandler(h)
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(run_dir / "run.log", encoding="utf-8")):
        h.setFormatter(fmt)
        root.addHandler(h)
    for noisy in ("matplotlib", "shap", "PIL", "interpret"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    with open(run_dir / "config.resolved.yaml", "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)
    versions = {}
    for pkg in PACKAGES:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = None
    env = {"python": sys.version.split()[0], "packages": versions, "config_hash": config_hash(cfg)}
    try:
        env["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                                    stderr=subprocess.DEVNULL).strip()
    except Exception:
        env["git_commit"] = None
    (run_dir / "environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    return run_dir


def _data_tables(ds: Dataset, train, test, rep: reporting.Reporter):
    """Tables that describe the data: cleaning flow, target distribution by partition (and stratum), features with
    missingness and binary prevalence, and, for categorical features, level concentration, overlap between
    partitions and per-level prevalence.
    """
    rep.table(pd.DataFrame(ds.flow), "data_flow", "Data cleaning flow", digits=0)
    y, t = ds.y_report, ds.times
    q = [0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1]
    rows = []
    strata = ds.strata_values()
    groups = [("all", None)] + ([(str(s), s) for s in pd.unique(strata)] if strata is not None else [])
    for stratum, sv in groups:
        for label, idx in (("train", train), ("test", test)):
            if sv is not None:
                idx = idx[strata[idx] == sv]
            if not len(idx):
                continue
            v = y[idx]
            rows.append({"partition": label, **({"stratum": stratum} if strata is not None else {}), "n": len(idx),
                         "start": pd.Timestamp(t[idx].min()).date(), "end": pd.Timestamp(t[idx].max()).date(),
                         "mean": v.mean(), "std": v.std(), **{f"q{int(x * 100)}": np.quantile(v, x) for x in q}})
    rep.table(pd.DataFrame(rows), "target_distribution", f"Target distribution by partition ({ds.target_unit})")

    X = ds.X
    feat_rows, level_rows, prev_rows = [], [], []
    for c in ds.features:
        vals = X[c]
        binary = vals.dropna().nunique() == 2 and c not in ds.categorical
        feat_rows.append({"feature": c, "type": "categorical" if c in ds.categorical else str(vals.dtype),
                          "missing_train_pct": vals.iloc[train].isna().mean() * 100,
                          "missing_test_pct": vals.iloc[test].isna().mean() * 100,
                          "prevalence_train_pct": (vals.iloc[train] == vals.max()).mean() * 100 if binary else np.nan,
                          "prevalence_test_pct": (vals.iloc[test] == vals.max()).mean() * 100 if binary else np.nan})
        if c in ds.categorical:
            tr, te = vals.iloc[train].value_counts(), vals.iloc[test].value_counts()
            share = vals.value_counts(normalize=True) * 100
            level_rows.append({"feature": c, "levels": vals.nunique(),
                               "top5_pct": share.head(5).sum(), "top10_pct": share.head(10).sum(),
                               "top15_pct": share.head(15).sum(), "train_only": len(tr.index.difference(te.index)),
                               "test_only": len(te.index.difference(tr.index)),
                               "common": len(tr.index.intersection(te.index)), "train_levels_lt10": int((tr < 10).sum()),
                               "test_rows_unseen_pct": vals.iloc[test].isin(te.index.difference(tr.index)).mean() * 100})
            if vals.nunique() <= MAX_PREVALENCE_LEVELS:
                p_tr = vals.iloc[train].value_counts(normalize=True, dropna=False) * 100
                p_te = vals.iloc[test].value_counts(normalize=True, dropna=False) * 100
                for level in p_tr.index.union(p_te.index):
                    prev_rows.append({"feature": c, "level": str(level), "train_pct": p_tr.get(level, 0.0),
                                      "test_pct": p_te.get(level, 0.0)})
    rep.table(pd.DataFrame(feat_rows), "features", "Feature set, missingness and binary prevalence by partition", 2)
    if level_rows:
        rep.table(pd.DataFrame(level_rows), "categorical_levels",
                  "Categorical levels: concentration (share of all rows held by the most frequent levels) and overlap "
                  "between partitions", 2)
    if prev_rows:
        rep.table(pd.DataFrame(prev_rows).sort_values(["feature", "train_pct"], ascending=[True, False]),
                  "categorical_prevalence", f"Prevalence of every level of the categorical features with at most "
                  f"{MAX_PREVALENCE_LEVELS} levels, by partition (%)", 2)


def deployable(ds: Dataset, res: StackResult, cfg: dict, meta_name: str) -> StackedPipeline:
    """Deployable pipeline of the selected base learners and the combiner ``meta_name``, returning report-scale
    predictions with the combiner's smearing factors."""
    factors = ReportScale(ds, res, cfg).factors.get(meta_name)
    return StackedPipeline([res.bases[b].model for b in res.selected], res.metas[meta_name].model, ds.features,
                           cfg["evaluation"]["clip_min"], meta_name, ds.transform, factors, ds.strata_col)


class ReportScale:
    """Converts the model-scale predictions of one stack to the report scale.

    Smearing factors come from out-of-sample residuals on the training partition only: out-of-fold
    predictions for base learners, selection-window predictions for combiners.
    """

    def __init__(self, ds: Dataset, res: StackResult, cfg: dict):
        self.ds, self.res = ds, res
        self.ts, self.clip = ds.transform, cfg["evaluation"]["clip_min"]
        self.strata = ds.strata_values()
        mode = cfg["dataset"]["target_smearing"]
        self.factors: dict[str, dict] = {}
        if self.ts.kind != "none":
            y = ds.y
            s_tr = None if self.strata is None else self.strata[res.train]
            for name, b in res.bases.items():
                self.factors[name] = smearing_factors(y[res.train], b.oof, s_tr, mode)
            if res.meta_rows is not None:
                rows = res.train[res.meta_rows]
                for name, m in res.metas.items():
                    if m.oos is not None:
                        self.factors[name] = smearing_factors(
                            y[rows], m.oos, None if self.strata is None else self.strata[rows], mode)

    def convert(self, name: str, p_model: np.ndarray, rows: np.ndarray) -> np.ndarray:
        """Report-scale predictions of model ``name`` for the rows ``rows``."""
        strata = None if self.strata is None else self.strata[rows]
        return to_report(p_model, self.ts, self.factors.get(name), strata, self.clip)

    def test(self) -> dict[str, np.ndarray]:
        """Report-scale test predictions of every base learner and combiner."""
        return {k: self.convert(k, v, self.res.test) for k, v in self.res.test_predictions().items()}

    def oof(self, name: str) -> np.ndarray:
        """Report-scale out-of-fold predictions of base learner ``name``."""
        return self.convert(name, self.res.bases[name].oof, self.res.train)

    def table(self) -> pd.DataFrame:
        """Smearing factors per model and stratum, as a table."""
        return pd.DataFrame([{"model": m, "stratum": k, "factor": v} for m, f in self.factors.items()
                             for k, v in f.items()])


def select_combiner(selection: StackResult) -> str:
    """The combiner with the lowest selection score on the meta-dataset of ``selection``."""
    return min(selection.metas.values(), key=lambda m: m.selection_score).name


def _evaluate(ds: Dataset, res: StackResult, cfg: dict, rep: reporting.Reporter, run_dir: Path,
              selection: StackResult | None = None) -> dict:
    """Evaluate the main stack on the test partition and write tables, figures and predictions.

    Covers the base learners out of fold and on test, performance per fold, the main comparison, combiner selection,
    learned weights, correlations of the base predictions, paired statistical tests, duration bands (overall and
    within each stratum), and performance by stratum, by training frequency of categorical levels and by calendar
    period. ``selection`` is the stack whose out-of-fold predictions choose the combiner (``res`` if not given).
    Returns the reference learner, the selected combiner and the scheme it was selected on.
    """
    ev, clip = cfg["evaluation"], cfg["evaluation"]["clip_min"]
    y, train, test = ds.y_report, res.train, res.test
    y_tr, y_te = y[train], y[test]
    scale = ReportScale(ds, res, cfg)
    preds = scale.test()
    oofs = {name: scale.oof(name) for name in res.bases}
    ref = res.selected[0]
    selection = selection or res
    best_meta = select_combiner(selection)
    if scale.factors:
        rep.table(scale.table(), "target_smearing",
                  f"Retransformation ({ds.transform.kind} target): smearing factor per model and stratum, "
                  "estimated on out-of-sample training predictions", 4, summary=False)

    rows = []
    for name, b in res.bases.items():
        m = np.isfinite(oofs[name])
        rows.append({"model": name, "partition": "train (OOF)", **metrics.compute(y_tr[m], oofs[name][m])})
        rows.append({"model": name, "partition": "test", **metrics.compute(y_te, preds[name])})
    base_tab = pd.DataFrame(rows)
    order = base_tab[base_tab.partition == "train (OOF)"].sort_values("RMSE").model.tolist()
    base_tab = base_tab.set_index("model").loc[order].reset_index()
    rep.table(base_tab, "base_learners_oof_vs_test",
              "Base learners: training-partition out-of-fold vs. chronological test performance")

    fold_rows = []
    for name, b in res.bases.items():
        for i, f in enumerate(res.folds):
            pos = np.searchsorted(train, f.valid)
            p = oofs[name][pos]
            if not np.isfinite(p).any():
                continue
            fp = b.fold_params[i] if i < len(b.fold_params) else {}
            fold_rows.append({"model": name, "fold": i, "valid_start": pd.Timestamp(ds.times[f.valid].min()).date(),
                              "valid_end": pd.Timestamp(ds.times[f.valid].max()).date(),
                              **metrics.compute(y[f.valid], p),
                              "n_estimators": fp.get("n_estimators", np.nan)})
    rep.table(pd.DataFrame(fold_rows), "oof_by_fold",
              "Out-of-fold performance of every base learner per held-out block (training partition)", 3)

    main = metrics.table(y_te, preds)
    main["role"] = ["selected base" if k in res.selected else ("ensemble" if k in res.metas else "base") for k in main.model]
    main["validation_score"] = [selection.metas[k].selection_score if k in selection.metas else np.nan
                                for k in main.model]
    rank = {k: i for i, k in enumerate(res.selected)}
    main = pd.concat([main[main.role == "selected base"].sort_values("model", key=lambda s: s.map(rank)),
                      main[main.role == "ensemble"], main[main.role == "base"]])
    rep.table(main, "test_performance", f"Test-partition performance (reference: {ref}; selected ensemble: {best_meta})")

    sel_rows = []
    meta_times = ds.times[selection.train[selection.meta_rows]]
    for name, m in selection.metas.items():
        for r in m.selection_windows:
            sel_rows.append({"combiner": name, "window": r["window"],
                             "valid_start": pd.Timestamp(meta_times[r["start_row"]]).date(),
                             "valid_end": pd.Timestamp(meta_times[r["end_row"] - 1]).date(),
                             "fit_rows": r["fit_rows"], "n": r["n"], "score": r["score"]})
    if sel_rows:
        sel = pd.DataFrame(sel_rows)
        sel["rank_in_window"] = sel.groupby("window")["score"].rank(method="min").astype(int)
        rep.table(sel, "combiner_selection_windows",
                  f"Combiner selection ({cfg['stacking']['meta_selection']}, {selection.scheme} out-of-fold "
                  f"predictions): {cfg['evaluation']['primary_metric']} per selection window; each combiner fitted "
                  "and tuned on earlier meta-rows only", 4, summary=False)
        agg = sel.groupby("combiner").agg(selection_score=("score", "mean"), score_std=("score", "std"),
                                          mean_rank=("rank_in_window", "mean"),
                                          windows_ranked_first=("rank_in_window", lambda x: int((x == 1).sum())))
        agg["selected"] = agg.index == best_meta
        rep.table(agg.reset_index().sort_values("selection_score"), "combiner_selection",
                  f"Combiner selection summary over {sel.window.nunique()} window(s) of the {selection.scheme} "
                  "out-of-fold predictions: mean score (the selection criterion), its spread, mean rank and number "
                  "of windows ranked first", 4)

    wrows = []
    for name, m in res.metas.items():
        w = getattr(m.model, "weights", None)
        if w is not None:
            wrows.append({"ensemble": name, **dict(zip(res.selected, np.round(w, 4)))})
    if wrows:
        rep.table(pd.DataFrame(wrows), "ensemble_weights", "Learned weights of the linear combiners", 4)

    Z = res.oof_matrix()[res.meta_rows]
    E = Z - y_tr[res.meta_rows, None]
    corr = pd.concat([pd.DataFrame(np.corrcoef(Z.T), index=res.selected, columns=res.selected).assign(kind="prediction"),
                      pd.DataFrame(np.corrcoef(E.T), index=res.selected, columns=res.selected).assign(kind="error")])
    rep.table(corr.reset_index().rename(columns={"index": "model"}), "oof_correlation",
              "Correlation of out-of-fold predictions and errors among selected base learners")

    compare = {ref: preds[ref]} | {k: preds[k] for k in res.selected[1:]} | {k: preds[k] for k in res.metas}
    strata = ds.frame[ds.strata_col].to_numpy()[test] if ds.strata_col else None
    clusters = ds.frame[ds.cluster_col].to_numpy()[test] if ds.cluster_col else None
    st_cfg = cfg["statistics"]
    frames = [paired_comparison(y_te, compare, ref, clusters, strata, st_cfg["alpha"],
                                st_cfg["bootstrap_resamples"], cfg["run"]["seed"])]
    for extra in st_cfg["extra_references"]:
        if extra == ref or extra not in preds:
            continue
        comp = {extra: preds[extra]} | {k: preds[k] for k in res.metas}
        frames.append(paired_comparison(y_te, comp, extra, clusters, strata, st_cfg["alpha"],
                                        st_cfg["bootstrap_resamples"], cfg["run"]["seed"]))
    st = pd.concat(frames, ignore_index=True)
    rep.table(st, "statistical_tests",
              "Paired comparison against the reference base learner and any additional references (Wilcoxon, Holm "
              "within each reference; cluster-bootstrap 95% CI; Cohen's d_z)", 4)

    edges = np.quantile(y_tr, ev["band_quantiles"])
    labels = np.asarray(ev["band_labels"])
    focus = {k: preds[k] for k in [ref, *res.metas]}
    band_actual = labels[np.digitize(y_te, edges)]
    band_pred = labels[np.digitize(preds[ref], edges)]
    bands = pd.concat([metrics.by_group(y_te, focus, band_actual, "band").assign(stratified_by="actual"),
                       metrics.by_group(y_te, focus, band_pred, "band").assign(stratified_by=f"predicted ({ref})")])
    bands["band"] = pd.Categorical(bands["band"], labels)
    bands = bands.sort_values(["stratified_by", "band", "RMSE"])
    rep.table(bands, "duration_bands", "Test performance by target band (edges from training-partition quantiles "
              f"{', '.join(f'{e:.2f}' for e in edges)} {ds.target_unit})")
    if ds.strata_col:
        # with strata of very different scale (e.g. pickup and delivery) pooled bands mostly separate the strata;
        # here the band edges are the training quantiles of each stratum
        s_all = ds.strata_values()
        s_tr, s_te = s_all[train], s_all[test]
        frames, edge_rows = [], []
        for sv in pd.unique(s_te):
            m_tr, m_te = s_tr == sv, s_te == sv
            if not m_tr.any():
                continue
            e_s = np.quantile(y_tr[m_tr], ev["band_quantiles"])
            edge_rows.append(f"{sv}: {', '.join(f'{e:.2f}' for e in e_s)}")
            sub = {k: v[m_te] for k, v in focus.items()}
            for how, values in (("actual", y_te[m_te]), (f"predicted ({ref})", preds[ref][m_te])):
                g = labels[np.digitize(values, e_s)]
                frames.append(metrics.by_group(y_te[m_te], sub, g, "band").assign(stratum=str(sv), stratified_by=how))
        sb = pd.concat(frames, ignore_index=True)
        sb["band"] = pd.Categorical(sb["band"], labels)
        sb = sb[["stratum", "stratified_by", "band", *[c for c in sb.columns if c not in ("stratum", "stratified_by",
                                                                                        "band")]]]
        rep.table(sb.sort_values(["stratum", "stratified_by", "band", "RMSE"]), "duration_bands_by_stratum",
                  f"Test performance by target band within each stratum (edges from the training quantiles of the "
                  f"stratum, {ds.target_unit}: {'; '.join(edge_rows)})")

    focus_all = {k: preds[k] for k in [*res.selected, *st_cfg["extra_references"], *res.metas] if k in preds}
    if strata is not None:
        by_stratum = metrics.by_group(y_te, focus_all, np.asarray(strata), "stratum").sort_values(["stratum", "RMSE"])
        rep.table(by_stratum, "performance_by_stratum",
                  f"Test performance by stratum ('{ds.strata_col}'), for the selected base learners and all combiners")
        rows = []
        for sv in pd.unique(strata):
            m_tr = ds.frame[ds.strata_col].to_numpy()[train] == sv
            rows.append({"stratum": sv, "n_train": int(m_tr.sum()), "n_test": int((np.asarray(strata) == sv).sum()),
                         "target_mean_train": float(y_tr[m_tr].mean()) if m_tr.any() else np.nan,
                         "target_mean_test": float(y_te[np.asarray(strata) == sv].mean())})
        rep.table(pd.DataFrame(rows), "stratum_sizes", "Stratum sizes and target means by partition", 3, summary=False)

    freq_frames = []
    thr = ev["rare_threshold"]
    for c in ds.categorical:
        counts = ds.frame[c].iloc[train].astype(object).value_counts()
        vals = ds.frame[c].iloc[test].astype(object)
        n_seen = vals.map(counts).fillna(0).to_numpy()
        group = np.where(vals.isna().to_numpy(), "missing",
                         np.where(n_seen == 0, "unseen", np.where(n_seen < thr, f"rare (<{thr})", "common")))
        t = metrics.by_group(y_te, focus_all, group, "level_frequency").assign(feature=c)
        t["share_of_test_pct"] = t["n"] / len(y_te) * 100
        freq_frames.append(t)
    if freq_frames:
        freq = pd.concat(freq_frames, ignore_index=True)
        freq = freq[["feature", "level_frequency", *[k for k in freq.columns if k not in ("feature", "level_frequency")]]]
        rep.table(freq.sort_values(["feature", "level_frequency", "RMSE"]), "performance_by_category_frequency",
                  "Test performance by training-partition frequency of each categorical level "
                  "(unseen levels are the cold-start case)", 3)

    period = pd.to_datetime(ds.times[test]).to_period(ev["period"]).astype(str)
    per = metrics.by_group(y_te, focus, np.asarray(period), "period").sort_values(["period", "model"])
    rep.table(per, "performance_by_period", "Test performance by calendar period", summary=False)
    reporting.metric_over_time(per, "period", "RMSE", rep.figure_path("rmse_by_period"), "Test RMSE by period")

    for name in (ref, best_meta):
        reporting.actual_vs_predicted(y_te, preds[name], rep.figure_path(f"actual_vs_predicted_{name}"), name, ds.target_unit)
        reporting.residual_histogram(y_te, preds[name], rep.figure_path(f"residuals_{name}"), name, ds.target_unit)

    # grouping columns are stored next to the predictions, so that any later per-group analysis (per phase, per
    # delivery company, per city) needs no recomputation; names that clash with a model get a "group:" prefix
    keep = list(dict.fromkeys(c for c in (ds.strata_col, ds.cluster_col, ds.group_col, *ds.categorical) if c))

    def groups(rows, taken):
        """Grouping columns of ``rows``; a name that clashes with a model in ``taken`` gets a ``group:`` prefix."""
        return {(f"group:{c}" if c in taken else c): ds.frame[c].to_numpy()[rows] for c in keep}

    pd.DataFrame({"time": ds.times[test], "y": y_te, **preds, **groups(test, preds)}).to_parquet(
        run_dir / "predictions" / "test.parquet")
    pd.DataFrame({"time": ds.times[train], "y": y_tr, **oofs, **groups(train, oofs)}).to_parquet(
        run_dir / "predictions" / "oof.parquet")
    return {"reference": ref, "best_meta": best_meta, "selection_scheme": selection.scheme}


def _tuning_tables(ds: Dataset, res: StackResult, cfg: dict, rep: reporting.Reporter):
    """Tables of the hyperparameters selected in every search, the search spaces and the training cost."""
    rows = [r for b in res.bases.values() for r in b.tuning]
    tab = pd.DataFrame(rows)
    t = ds.times
    for col in ("inner_fit_start", "inner_fit_end", "inner_val_start", "inner_val_end"):
        if col in tab:
            tab[col] = [pd.Timestamp(t[int(v)]) if pd.notna(v) else pd.NaT for v in tab[col]]
    metas = pd.DataFrame([m.tuning for m in res.metas.values() if m.tuning])
    metas = metas.drop(columns=[c for c in ("inner_fit_start", "inner_fit_end", "inner_val_start", "inner_val_end")
                                if c in metas], errors="ignore")
    rows = pd.concat([tab, metas], ignore_index=True)
    rep.table(rows, "hyperparameters", "Selected hyperparameters per model and nested split", 4, summary=False)
    rep.table(space_table(cfg["tuning"]), "search_spaces", "Hyperparameter search spaces", 4, summary=False)
    timing = [{"model": k, **b.timing} for k, b in res.bases.items()] + \
             [{"model": k, **m.timing} for k, m in res.metas.items()]
    rep.table(pd.DataFrame(timing), "training_cost", "Wall-clock training cost (seconds)", 2)


def _scheme_table(ds: Dataset, stacks: dict[str, StackResult], cfg: dict, rep: reporting.Reporter):
    """Test performance of the selected base learners and the combiners under every out-of-fold scheme."""
    frames = []
    for scheme, res in stacks.items():
        keep = res.selected + list(res.metas)
        p = ReportScale(ds, res, cfg).test()
        frames.append(metrics.table(ds.y_report[res.test], {k: p[k] for k in keep}, scheme=scheme,
                                    meta_rows=len(res.meta_rows), selected="+".join(res.selected)))
    rep.table(pd.concat(frames), "oof_scheme_comparison",
              "Sensitivity to the out-of-fold scheme (blocked K-fold vs. forward-chaining)")


# Base learners with exact TreeSHAP support.
TREE_LEARNERS = ("LGBM", "XGB", "CAT")
MAX_PREVALENCE_LEVELS = 20   # categorical features with at most this many levels get a per-level prevalence table


def shap_seed_model(selected: list[str]) -> str | None:
    """The reference learner if it supports TreeSHAP, otherwise the first selected tree learner, or ``None``."""
    return next((m for m in selected if m in TREE_LEARNERS), None)


def _robustness(ds, res, cfg, cache, rep, explain_idx):
    """Refit the selected base learners and every combiner with the other seeds of ``robustness.seeds``, keeping all
    hyperparameters fixed, and report test performance per seed, its mean and spread, and the stability of the
    TreeSHAP importance ranking across seeds.
    """
    seeds = [s for s in cfg["robustness"]["seeds"] if s != cfg["run"]["seed"]]
    frozen = frozen_params(res)
    frozen["base"] = {k: v for k, v in frozen["base"].items() if k in res.selected}
    keep = res.selected + list(res.metas)
    y_te = ds.y_report[res.test]
    frames = [metrics.table(y_te, {k: v for k, v in ReportScale(ds, res, cfg).test().items() if k in keep},
                            seed=res.seed)]
    importances = {}
    shap_model = shap_seed_model(res.selected)
    if shap_model:
        importances[res.seed] = explain.importance(explain.tree_shap(res.bases[shap_model].model,
                                                                     ds.X.iloc[explain_idx]))
    for s in seeds:
        r = run_stack(ds, res.train, res.test, cfg, res.scheme, res.selected, s, "none",
                      keep_models=shap_model is not None, fixed=frozen, selected=res.selected, cache=cache,
                      tag="seeds")
        frames.append(metrics.table(y_te, {k: v for k, v in ReportScale(ds, r, cfg).test().items() if k in keep},
                                    seed=s))
        if shap_model and r.bases[shap_model].model is not None:
            importances[s] = explain.importance(explain.tree_shap(r.bases[shap_model].model, ds.X.iloc[explain_idx]))
    per_seed = pd.concat(frames)
    rep.table(per_seed, "seed_runs", "Test performance per random seed (hyperparameters fixed)", summary=False)
    cols = ["RMSE", "MAE", "MAPE", "SMAPE", "R2"]
    agg = per_seed.groupby("model")[cols].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    rep.table(agg.reset_index().sort_values("RMSE_mean"), "seed_robustness",
              f"Seed robustness: mean and standard deviation over {per_seed.seed.nunique()} seeds", 4)
    if len(importances) > 1:
        rep.table(explain.seed_stability(importances).assign(model=shap_model), "shap_seed_stability",
                  f"Stability of {shap_model} TreeSHAP importance rankings across seeds "
                  f"({'reference learner' if shap_model == res.selected[0] else 'first selected tree learner'})", 3)


def rolling_learners(spec, selected: list[str], candidates: list[str]) -> list[str]:
    """``selected``: the model families chosen in the main experiment, re-tuned and refitted at every origin;
    ``all``: every candidate, with top-k selection repeated at every origin; or an explicit list of additional
    families, combined with the selected ones."""
    if spec == "selected":
        return list(selected)
    if spec == "all":
        return list(candidates)
    # an explicit list is added to the families selected in the main experiment, so that the fixed reference is
    # always available and top-k selection is repeated over the enlarged candidate set at every origin
    return list(dict.fromkeys([*selected, *spec]))


def rolling_reference(spec: str, main_reference: str, origin_selected: list[str]) -> str:
    """``main``: the reference base learner of the main experiment at every origin (a fixed comparator);
    ``selected``: the best out-of-fold base learner of each origin; or an explicit learner name."""
    if spec == "main":
        return main_reference
    if spec == "selected":
        return origin_selected[0]
    return spec


def _rolling(ds, cfg, cache, rep, selected):
    """Rolling-origin evaluation, separate from the main experiment.

    At every origin the complete pipeline (tuning with one search per model family, out-of-fold predictions,
    base-learner selection, combiners and combiner selection) is rebuilt from the earlier rows and evaluated on the
    following window. Reports performance per origin, the learners selected at each origin, the protocol-selected
    combiner compared with the reference and with the best single learner in hindsight, and a summary over the
    origins.
    """
    r = cfg["rolling"]
    learners = rolling_learners(r["base_learners"], selected, cfg["base_learners"])
    fixed_ref = rolling_reference(r["reference"], selected[0], selected)
    if r["reference"] != "selected" and fixed_ref not in learners:
        raise ValueError(f"rolling.reference '{fixed_ref}' is not among the rolling base learners {learners}")
    rcfg = {**cfg, "stacking": {**cfg["stacking"], "top_k": min(cfg["stacking"]["top_k"], len(learners))}}
    origins = rolling_origins(len(ds.frame), ds.unit_starts(), r["n_origins"], r["test_fraction"], r["window"])
    rep.table(describe_folds(origins, ds.times, "rolling"), "rolling_windows", "Rolling-origin evaluation windows", 0)
    frames, chosen, comparison = [], [], []
    primary, sel_scheme = cfg["stacking"]["primary_scheme"], cfg["stacking"]["combiner_selection"]
    for j, f in enumerate(origins):
        res = run_stack(ds, f.train, f.valid, rcfg, primary, learners,
                        cfg["run"]["seed"], r["tuning_mode"], r["trial_scale"], cache=cache, tag=f"rolling{j}")
        selection = res
        if sel_scheme != primary:
            selection = run_stack(ds, f.train, f.valid, rcfg, sel_scheme, res.selected, cfg["run"]["seed"],
                                  r["tuning_mode"], r["trial_scale"], selected=res.selected, reuse_final=res.bases,
                                  cache=cache, tag=f"rolling{j}")
        chosen.append({"origin": j, "candidates": "+".join(learners), "selected": "+".join(res.selected),
                       "matches_main_experiment": sorted(res.selected) == sorted(selected)})
        p = ReportScale(ds, res, cfg).test()
        best_meta = select_combiner(selection)
        ref = rolling_reference(r["reference"], selected[0], res.selected)
        tab = metrics.table(ds.y_report[f.valid], p, origin=j, test_start=pd.Timestamp(ds.times[f.valid].min()).date())
        tab["kind"] = np.where(tab.model.isin(list(res.metas)), "ensemble", "base")
        tab["role"] = np.where(tab.model == ref, "reference",
                               np.where(tab.model == best_meta, "selected ensemble", ""))
        frames.append(tab)
        bases = tab[tab.kind == "base"].set_index("model")["RMSE"]
        rm = tab.set_index("model")["RMSE"]
        comparison.append({"origin": j, "test_start": tab.test_start.iloc[0], "reference": ref,
                           "reference_RMSE": rm[ref], "best_single_ex_post": bases.idxmin(),
                           "best_single_RMSE": bases.min(), "selected_ensemble": best_meta,
                           "selected_ensemble_RMSE": rm[best_meta],
                           "ensemble_minus_reference": rm[best_meta] - rm[ref],
                           "ensemble_minus_best_single": rm[best_meta] - bases.min()})
    roll = pd.concat(frames)
    rep.table(roll, "rolling_origin", "Rolling-origin evaluation: performance per origin", summary=False)
    rep.table(pd.DataFrame(chosen), "rolling_selection",
              f"Base learners selected at each rolling origin (main experiment: {'+'.join(selected)})", 0)
    rep.table(pd.DataFrame(comparison), "rolling_comparison",
              "Rolling origin: protocol-selected ensemble against the reference and against the best single base "
              "learner in hindsight (RMSE; negative differences favour the ensemble)", 4)
    cols = ["RMSE", "MAE", "MAPE", "SMAPE", "R2"]
    agg = roll.groupby("model")[cols].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    beats_ref, beats_best, ranks = [], [], []
    for j, g in roll.groupby("origin"):
        rm = g.set_index("model")["RMSE"]
        ref_val = g.loc[g.role == "reference", "RMSE"].iloc[0]
        best_single = g.loc[g.kind == "base", "RMSE"].min()
        beats_ref.append(rm < ref_val)
        beats_best.append(rm < best_single)
        ranks.append(rm.rank())
    agg["origins_better_than_reference"] = pd.concat(beats_ref, axis=1).sum(axis=1)
    agg["origins_better_than_best_single"] = pd.concat(beats_best, axis=1).sum(axis=1)
    agg["mean_rank"] = pd.concat(ranks, axis=1).mean(axis=1)
    label = f"fixed reference {fixed_ref}" if r["reference"] != "selected" else "per-origin reference"
    rep.table(agg.reset_index().sort_values("RMSE_mean"), "rolling_origin_summary",
              f"Rolling-origin summary over {len(origins)} origins (mean, std; {label}; count of origins beating "
              "the reference and the best single base learner in hindsight; mean RMSE rank)", 4)
    reporting.metric_over_time(roll[roll.model.isin(roll.loc[roll.role != "", "model"].unique())], "origin", "RMSE",
                               rep.figure_path("rolling_origin_rmse"), "Rolling-origin RMSE")


def _explain(ds, res, cfg, rep, info, explain_idx) -> StackedPipeline:
    """Three-level SHAP analysis of the main stack on a sample of test rows.

    TreeSHAP of the selected gradient-boosting base learners, attribution of every combiner's output to the base
    predictions, and permutation SHAP of the complete deployed pipeline, together with the agreement of their
    importance rankings. Explanations are stored so that their figures can be redrawn. Returns the deployed
    pipeline.
    """
    e = cfg["explain"]
    rng = np.random.default_rng(cfg["run"]["seed"])
    X_s = ds.X.iloc[explain_idx]
    ref_pred = res.bases[res.selected[0]].model.predict(X_s)
    wf = int(np.argsort(ref_pred)[int(e["waterfall_quantile"] * (len(ref_pred) - 1))])
    tree_names = [n for n in (res.selected if e["tree_models"] == "selected" else e["tree_models"])
                  if n in TREE_LEARNERS and res.bases.get(n) and res.bases[n].model is not None]
    if not tree_names and e["tree_models"] == "selected":
        # no gradient-boosting learner was selected: explain the most accurate one out of fold, so that a
        # feature-level TreeSHAP view of the original variables is always available
        y_tr = ds.y[res.train]
        avail = [n for n in TREE_LEARNERS if res.bases.get(n) and res.bases[n].model is not None]
        if avail:
            best_tree = min(avail, key=lambda n: metrics.rmse(y_tr[np.isfinite(res.bases[n].oof)],
                                                              res.bases[n].oof[np.isfinite(res.bases[n].oof)]))
            tree_names = [best_tree]
            rep.text(f"No gradient-boosting learner was selected; Level-0 TreeSHAP is shown for {best_tree}, the "
                     "most accurate one out of fold.")
    imps = {}
    labels, values = e.get("feature_labels") or {}, e.get("value_labels") or {}
    store = Path(rep.figures).parent / "explanations"
    store.mkdir(exist_ok=True)

    def draw(ex, name, **args):
        """Store an explanation and draw its figures; meta-level figures show base learners under their display
        names.
        """
        explain.save_explanation(store / f"{name}.joblib", ex, **args)
        shown = {**labels, **{m: reporting.model_label(m) for m in res.selected}} if name.startswith("shap_meta") \
            else labels
        explain.plot_explanation(ex, rep.figure_path(name), labels=shown, value_labels=values, **args)

    for n in tree_names:
        ex = explain.tree_shap(res.bases[n].model, X_s)
        draw(ex, f"shap_level0_{n}", dependence_top=e["dependence_top"], waterfall_index=wf)
        imps[f"Level-0 {n}"] = explain.importance(ex)

    pos = np.searchsorted(res.test, explain_idx)
    Zs = np.column_stack([res.bases[b].test[pos] for b in res.selected])
    Z_ref = res.oof_matrix()[res.meta_rows]
    meta_rows = []
    for name, m in res.metas.items():
        ex = explain.meta_shap(m.model, Zs, res.selected, Z_ref)
        imp = explain.importance(ex).reindex(res.selected)
        meta_rows.append({"ensemble": name, **imp.to_dict(), **{f"{k}_share": v / imp.sum() for k, v in imp.items()}})
        draw(ex, f"shap_meta_{name}", dependence_top=0, waterfall_index=None, max_display=len(res.selected))
    rep.table(pd.DataFrame(meta_rows), "shap_meta_level", "Meta-level attribution: mean |SHAP| of each base prediction", 3)

    best = res.metas[info["best_meta"]]
    pipe = deployable(ds, res, cfg, best.name)
    encoder = TabularEncoder(ds.features, ds.categorical).fit(ds.X.iloc[res.train])
    X_exp = X_s.iloc[: e["pipeline_shap_rows"]]
    X_bg = ds.X.iloc[rng.choice(res.train, e["background_rows"], replace=False)]
    t0 = time.perf_counter()
    ex_full = explain.pipeline_shap(pipe, encoder, X_exp, X_bg, e["pipeline_permutations"], cfg["run"]["seed"])
    log.info("xai  | composite permutation SHAP on %d rows in %.1fs", len(X_exp), time.perf_counter() - t0)
    wf_exp = int(np.argsort(ref_pred[: len(X_exp)])[int(e["waterfall_quantile"] * (len(X_exp) - 1))])
    draw(ex_full, f"shap_pipeline_{best.name}", dependence_top=e["dependence_top"], waterfall_index=wf_exp)
    imps[f"Pipeline {best.name}"] = explain.importance(ex_full)

    imp_tab = pd.DataFrame(imps).sort_values(f"Pipeline {best.name}", ascending=False)
    imp_tab = imp_tab.reset_index().rename(columns={"index": "feature"})
    if labels:
        imp_tab.insert(1, "label", imp_tab["feature"].map(lambda f: labels.get(f, f)))
    rep.table(imp_tab, "shap_feature_importance",
              "Global feature importance (mean |SHAP|) at Level-0 and for the full pipeline", 3)
    same = {f"Level-0 {n}": explain.importance(explain.tree_shap(res.bases[n].model, X_exp)) for n in tree_names}
    same[f"Pipeline {best.name}"] = imps[f"Pipeline {best.name}"]
    agree = []
    names = list(same)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            agree.append({"a": a, "b": b, "rows": len(X_exp), **explain.agreement(same[a], same[b], e["top_k_overlap"])})
    rep.table(pd.DataFrame(agree), "shap_agreement",
              "Agreement of importance rankings on identical instances (Level-0 proxies vs. full pipeline)", 3)
    return pipe


def redraw(run_dir: Path, config: str | None = None):
    """Redraw the SHAP figures and the prediction plots of a finished run from its stored results."""
    import joblib
    with open(run_dir / "config.resolved.yaml", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if config:
        from .config import load_config
        new = load_config(config)["explain"]
        cfg["explain"]["feature_labels"], cfg["explain"]["value_labels"] = new["feature_labels"], new["value_labels"]
    labels = cfg["explain"].get("feature_labels") or {}
    values = cfg["explain"].get("value_labels") or {}
    figures = run_dir / "figures"
    for path in sorted((run_dir / "explanations").glob("*.joblib")):
        item = joblib.load(path)
        ex = item["explanation"]
        shown = {**labels, **{m: reporting.model_label(m) for m in ex.feature_names}} \
            if path.stem.startswith("shap_meta") else labels
        explain.plot_explanation(ex, str(figures / path.stem), labels=shown, value_labels=values, **item["plot_args"])
    per = run_dir / "tables" / "performance_by_period.csv"
    if per.exists():
        reporting.metric_over_time(pd.read_csv(per), "period", "RMSE", str(figures / "rmse_by_period"))
    roll = run_dir / "tables" / "rolling_origin.csv"
    if roll.exists():
        r = pd.read_csv(roll)
        reporting.metric_over_time(r[r.model.isin(r.loc[r.role.notna() & (r.role != ""), "model"].unique())],
                                   "origin", "RMSE", str(figures / "rolling_origin_rmse"))
    pred = run_dir / "predictions" / "test.parquet"
    if pred.exists():
        df = pd.read_parquet(pred)
        unit = cfg["dataset"].get("target_unit", "")
        for name in [c for c in df.columns if c not in ("time", "y")]:
            if (figures / f"actual_vs_predicted_{name}.png").exists():
                reporting.actual_vs_predicted(df["y"].to_numpy(), df[name].to_numpy(),
                                              str(figures / f"actual_vs_predicted_{name}"), name, unit)
                reporting.residual_histogram(df["y"].to_numpy(), df[name].to_numpy(),
                                             str(figures / f"residuals_{name}"), name, unit)
    log.info("plots | figures redrawn in %s", figures)


def run(cfg: dict, resume: bool = True) -> Path:
    """Run the complete experiment described by ``cfg`` and return its output directory.

    The data are loaded and the most recent rows held out (or the user-supplied test file is used). Out-of-fold
    predictions and combiners are produced under every configured scheme, the main stack is evaluated on the test
    partition, and the optional robustness, rolling-origin, sensitivity, explanation and profiling analyses follow.
    With ``run.stop_after: base_learners`` the run stops once the base learners are checkpointed. With ``resume``
    completed units are restored from their checkpoints.
    """
    run_dir = _setup(cfg)
    rep = reporting.Reporter(run_dir)
    cache = Cache(run_dir / "cache", resume)
    seed, s = cfg["run"]["seed"], cfg["stacking"]
    t_start = time.perf_counter()

    ds = load_dataset(cfg)
    if "_partition" in ds.frame:
        train, test, check = given_partitions(ds.frame["_partition"].to_numpy(), ds.times)
        if not check["chronological"]:
            log.warning("split | %s test rows precede the end of the training period; the evaluation is not "
                        "strictly out-of-time", f"{check['test_rows_before_train_end']:,}")
        rep.text(f"Partitions supplied by the user (`dataset.test_path`); chronological: {check['chronological']} "
                 f"({check['test_rows_before_train_end']:,} test rows precede the end of the training period).")
    else:
        train, test = holdout(len(ds.frame), ds.unit_starts(), cfg["split"]["test_fraction"])
    log.info("split | train=%s test=%s", f"{len(train):,}", f"{len(test):,}")
    _data_tables(ds, train, test, rep)

    primary = s["primary_scheme"]
    if cfg["run"]["stop_after"] == "base_learners":
        part = run_stack(ds, train, test, cfg, primary, cfg["base_learners"], seed, cfg["tuning"]["mode"],
                         cache=cache, base_only=True)
        stage_scale = ReportScale(ds, part, cfg)
        rows = []
        for k, b in part.bases.items():
            o = stage_scale.oof(k)
            rows.append({"model": k, **metrics.compute(ds.y_report[train][np.isfinite(o)], o[np.isfinite(o)]),
                         **b.timing})
        rep.table(pd.DataFrame(rows).sort_values("RMSE"), "stage_" + "_".join(part.bases).lower(),
                  "Staged run: out-of-fold performance and cost of the base learners trained in this stage", 3)
        rep.text(f"Stage finished after the base learners ({cache.misses} units computed, {cache.hits} restored). "
                 "Run the configuration listing all base learners to complete the experiment.")
        rep.write_summary(f"Stage report: {cfg['run']['name']}")
        log.info("stage | base learners %s checkpointed; stopping (run.stop_after)", ", ".join(part.bases))
        return run_dir

    stacks: dict[str, StackResult] = {}
    stacks[primary] = run_stack(ds, train, test, cfg, primary, cfg["base_learners"], seed, cfg["tuning"]["mode"],
                                keep_models=True, cache=cache)
    res = stacks[primary]
    if res.failed:
        rep.table(pd.DataFrame([{"model": k, "error": v} for k, v in res.failed.items()]), "failed_learners",
                  "Base learners excluded after a failure (run.continue_on_error)", 0)
    rep.table(describe_folds(res.folds, ds.times, primary), "folds", "Out-of-fold partitions of the training period", 0,
              summary=False)
    for scheme in s["oof_schemes"]:
        if scheme == primary:
            continue
        learners = res.selected if s["secondary_learners"] == "selected" else cfg["base_learners"]
        stacks[scheme] = run_stack(ds, train, test, cfg, scheme, learners, seed, cfg["tuning"]["mode"],
                                   reuse_final=res.bases, cache=cache)

    # the deployed combiner is fitted on the primary scheme; it may be chosen on another scheme's out-of-fold
    # predictions (stacking.combiner_selection), e.g. forward chaining, whose meta-rows were produced by models
    # that never saw any later row
    info = _evaluate(ds, res, cfg, rep, run_dir, selection=stacks[s["combiner_selection"]])
    rep.text(f"Combiner selected on the {info['selection_scheme']} out-of-fold predictions: {info['best_meta']}.")
    _tuning_tables(ds, res, cfg, rep)
    if len(stacks) > 1:
        _scheme_table(ds, stacks, cfg, rep)

    rng = np.random.default_rng(seed)
    explain_idx = np.sort(rng.choice(test, min(cfg["explain"]["tree_shap_rows"], len(test)), replace=False))
    if len(cfg["robustness"]["seeds"]) > 1:
        _robustness(ds, res, cfg, cache, rep, explain_idx)
    if cfg["rolling"]["enabled"]:
        _rolling(ds, cfg, cache, rep, res.selected)
    if cfg["sensitivity"]["enabled"]:
        from .sensitivity import run_sensitivity, summarise
        sens = run_sensitivity(ds, res, cfg, cache, ReportScale(ds, res, cfg))
        rep.table(sens, "hyperparameter_sensitivity",
                  "One-at-a-time hyperparameter sensitivity around the selected configuration (descriptive; "
                  "rounds re-determined by early stopping on the inner split, refit on the training partition)", 4,
                  summary=False)
        rep.table(summarise(sens), "hyperparameter_sensitivity_summary",
                  "Largest change in test RMSE per hyperparameter (one-at-a-time sensitivity)", 4)

    best = res.metas[info["best_meta"]]
    pipe = deployable(ds, res, cfg, best.name)
    if cfg["explain"]["enabled"]:
        pipe = _explain(ds, res, cfg, rep, info, explain_idx)
    joblib.dump(pipe, run_dir / "models" / "pipeline.joblib")
    joblib.dump({k: m.model for k, m in res.metas.items()}, run_dir / "models" / "meta_learners.joblib")
    ds.X.iloc[test[: cfg["profiling"]["batch_rows"]]].to_parquet(run_dir / "models" / "profiling_sample.parquet")

    if cfg["profiling"]["enabled"]:
        cmd = [sys.executable, "-m", "stackxai", "profile", "--run-dir", str(run_dir)]
        log.info("profile | launching fresh process: %s", " ".join(cmd))
        subprocess.run(cmd, check=True)
        t = run_dir / "tables"
        rep.table(pd.read_csv(t / "profiling_latency.csv"), "profiling_latency_summary",
                  f"Single-record inference latency per stage and end-to-end ({best.name}, 1 thread)", 3)
        rep.table(pd.read_csv(t / "profiling_throughput.csv"), "profiling_throughput_summary",
                  "Inference throughput (threads share one process; processes each load their own copy)", 1)
        if (t / "profiling_meta.csv").exists():
            rep.table(pd.read_csv(t / "profiling_meta.csv"), "profiling_meta_summary",
                      "Single-record latency of every Level-1 combiner and the implied end-to-end latency", 3)
        rep.table(pd.read_csv(t / "profiling_memory.csv"), "profiling_memory_summary", "Inference memory (MB)", 1)

    rep.text(f"Total wall-clock time: {(time.perf_counter() - t_start) / 60:.1f} min "
             f"({cache.hits} checkpointed units restored, {cache.misses} computed)")
    rep.write_summary(f"Experiment report: {cfg['run']['name']}")
    log.info("done | results in %s", run_dir)
    return run_dir
