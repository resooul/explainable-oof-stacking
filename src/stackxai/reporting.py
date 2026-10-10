"""Table and figure export."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _fmt(v, digits: int) -> str:
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return f"{v:,}"
    if isinstance(v, (float, np.floating)):
        if not np.isfinite(v):
            return "--"
        if v != 0 and (abs(v) < 10 ** -digits or abs(v) >= 1e6):
            return f"{v:.2e}"
        return f"{v:.{digits}f}"
    return str(v)


def to_latex(df: pd.DataFrame, caption: str, label: str, digits: int = 3) -> str:
    esc = lambda s: (str(s).replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%")
                     .replace("&", r"\&").replace("#", r"\#").replace("$", r"\$"))
    cols = " ".join(["l"] + ["r"] * (df.shape[1] - 1))
    lines = [r"\begin{table}[ht]", r"\centering", rf"\caption{{{esc(caption)}}}", rf"\label{{{label}}}",
             r"\resizebox{\textwidth}{!}{%", rf"\begin{{tabular}}{{{cols}}}", r"\toprule",
             " & ".join(esc(c) for c in df.columns) + r" \\", r"\midrule"]
    lines += [" & ".join(esc(_fmt(v, digits)) for v in row) + r" \\" for row in df.itertuples(index=False)]
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(lines)


def to_markdown(df: pd.DataFrame, digits: int = 3) -> str:
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join(["---"] * df.shape[1]) + "|"
    body = ["| " + " | ".join(_fmt(v, digits) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])


class Reporter:
    def __init__(self, run_dir: Path):
        self.tables = run_dir / "tables"
        self.figures = run_dir / "figures"
        self.tables.mkdir(parents=True, exist_ok=True)
        self.figures.mkdir(parents=True, exist_ok=True)
        self.sections: list[str] = []

    def table(self, df: pd.DataFrame, name: str, caption: str, digits: int = 3, summary: bool = True):
        df = df.reset_index(drop=True)
        df.to_csv(self.tables / f"{name}.csv", index=False)
        if df.empty:
            self.sections.append(f"### {caption}\n\nNo rows.\n")
            return
        (self.tables / f"{name}.tex").write_text(to_latex(df, caption, f"tab:{name}", digits), encoding="utf-8")
        if summary:
            self.sections.append(f"### {caption}\n\n`tables/{name}.csv`\n\n{to_markdown(df, digits)}\n")

    def text(self, s: str):
        self.sections.append(s + "\n")

    def figure_path(self, name: str) -> str:
        return str(self.figures / name)

    def write_summary(self, title: str):
        (self.tables.parent / "SUMMARY.md").write_text(f"# {title}\n\n" + "\n".join(self.sections), encoding="utf-8")


def save_fig(path: str):
    plt.savefig(path + ".png", dpi=200, bbox_inches="tight")
    plt.savefig(path + ".pdf", bbox_inches="tight")
    plt.close("all")


# Figures carry no title and no units in their axis labels: the caption and the text give both.

def actual_vs_predicted(y, p, path: str, title: str = "", unit: str = ""):
    lim = np.percentile(np.r_[y, p], 99.5)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    hb = ax.hexbin(y, p, gridsize=80, extent=(0, lim, 0, lim), bins="log", mincnt=1, cmap="viridis")
    ax.plot([0, lim], [0, lim], "w--", lw=1)
    ax.set(xlabel="Actual", ylabel="Predicted", xlim=(0, lim), ylim=(0, lim))
    fig.colorbar(hb, ax=ax, label="Log10 count")
    save_fig(path)


def residual_histogram(y, p, path: str, title: str = "", unit: str = ""):
    r = y - p
    lo, hi = np.percentile(r, [0.5, 99.5])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(r[(r >= lo) & (r <= hi)], bins=100, color="#c0504d", alpha=0.8)
    ax.axvline(0, color="k", ls="--", lw=1)
    ax.set(xlabel="Residual, actual - predicted", ylabel="Frequency")
    save_fig(path)


MODEL_LABELS = {"FTT": "FT-Transformer", "RESNET": "ResNet", "TABNET": "TabNet", "RIDGE": "Ridge"}


def model_label(name: str) -> str:
    """Display name of a model as written in the article (EN_AVG -> EN-AVG, FTT -> FT-Transformer)."""
    return MODEL_LABELS.get(name, name.replace("_", "-"))


def metric_over_time(df: pd.DataFrame, x: str, metric: str, path: str, title: str = ""):
    fig, ax = plt.subplots(figsize=(7, 4))
    for model, g in df.groupby("model"):
        xs = (g[x].astype(int) + 1).astype(str) if x == "origin" else g[x].astype(str)   # origins numbered from 1
        ax.plot(xs, g[metric], marker="o", label=model_label(model))
    ax.set(xlabel=x.replace("_", " ").capitalize(), ylabel=metric)
    if x != "origin":
        ax.tick_params(axis="x", rotation=45)
    ax.legend(fontsize=8)
    save_fig(path)
