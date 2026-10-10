"""Leakage-aware temporal splitting.

All functions operate on row positions of a frame that is already in temporal
order (see ``data.temporal_order``). Cut points are snapped to unit
boundaries so that rows of one group never fall on both sides of a split.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Fold:
    train: np.ndarray
    valid: np.ndarray


def snap(position: int, unit_starts: np.ndarray, n: int) -> int:
    if position >= n:
        return n
    i = np.searchsorted(unit_starts, position, side="left")
    return int(unit_starts[i]) if i < len(unit_starts) else n


def holdout(n: int, unit_starts: np.ndarray, test_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    cut = snap(int(round(n * (1 - test_fraction))), unit_starts, n)
    return np.arange(cut), np.arange(cut, n)


def given_partitions(labels: np.ndarray, times: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Training/test positions from user-supplied files, with a check of their temporal order."""
    train, test = np.flatnonzero(labels == "train"), np.flatnonzero(labels == "test")
    if not len(train) or not len(test):
        raise ValueError("Both the training and the test file must contain rows after cleaning")
    overlap = int((times[test] < times[train].max()).sum())
    return train, test, {"test_rows_before_train_end": overlap, "chronological": overlap == 0}


def _cuts(n: int, unit_starts: np.ndarray, k: int) -> list[int]:
    cuts = [0] + [snap(int(round(i * n / k)), unit_starts, n) for i in range(1, k)] + [n]
    if len(set(cuts)) != len(cuts):
        raise ValueError("Too few units for the requested number of folds")
    return cuts


def _purge(train: np.ndarray, valid: np.ndarray, times: np.ndarray | None, gap) -> np.ndarray:
    if gap is None or times is None:
        return train
    gap = pd.Timedelta(gap)
    lo, hi = times[valid].min() - gap, times[valid].max() + gap
    keep = (times[train] < lo) | (times[train] > hi)
    return train[keep]


def temporal_folds(positions: np.ndarray, unit_starts: np.ndarray, k: int, scheme: str,
                   times: np.ndarray | None = None, gap=None) -> list[Fold]:
    """Contiguous temporal blocks over ``positions``.

    ``blocked``: every block is held out once and predicted by a model fitted on
    all remaining blocks (each row receives exactly one out-of-fold prediction).
    ``forward``: block j is predicted by a model fitted on blocks 0..j-1 only
    (the first block receives no prediction).
    """
    n = len(positions)
    local_starts = np.searchsorted(positions, unit_starts[(unit_starts >= positions[0]) & (unit_starts <= positions[-1])])
    cuts = _cuts(n, local_starts, k)
    t = times[positions] if times is not None else None
    folds = []
    for j in range(k):
        valid = np.arange(cuts[j], cuts[j + 1])
        if scheme == "blocked":
            train = np.r_[np.arange(0, cuts[j]), np.arange(cuts[j + 1], n)]
        elif scheme == "forward":
            if j == 0:
                continue
            train = np.arange(0, cuts[j])
        else:
            raise ValueError(f"Unknown OOF scheme '{scheme}'")
        train = _purge(train, valid, t, gap)
        folds.append(Fold(positions[train], positions[valid]))
    return folds


def contiguous_segments(positions: np.ndarray) -> list[np.ndarray]:
    """Split sorted row positions into runs of consecutive positions."""
    positions = np.sort(positions)
    if not len(positions):
        return []
    breaks = np.flatnonzero(np.diff(positions) > 1) + 1
    return np.split(positions, breaks)


def inner_segment(train: np.ndarray, mode: str = "contiguous", min_rows: int = 0) -> np.ndarray:
    """Rows of ``train`` from which the inner tuning split is drawn.

    ``contiguous``: the most recent run of consecutive rows (optionally extended
    backwards by earlier runs while it holds fewer than ``min_rows`` rows, which
    re-introduces the gap and is therefore off by default). Under the blocked
    out-of-fold scheme the training index of an interior fold has a hole where the
    held-out block lies; drawing the inner split across that hole would fit on two
    periods separated in time and validate on a third, which distorts early stopping
    and the selected configuration. ``all``: every row of ``train`` (legacy behaviour).
    """
    train = np.sort(train)
    if mode == "all":
        return train
    if mode != "contiguous":
        raise ValueError(f"tuning.inner_split must be 'contiguous' or 'all', got '{mode}'")
    segments = contiguous_segments(train)
    chosen = [segments.pop()]
    while segments and sum(len(s) for s in chosen) < min_rows:
        chosen.insert(0, segments.pop())
    return np.concatenate(chosen)


def inner_split(train: np.ndarray, unit_starts: np.ndarray, val_fraction: float,
                max_rows: int | None = None, segment: str = "contiguous",
                max_val_rows: int | None = None, position: str = "after_cut") -> tuple[np.ndarray, np.ndarray]:
    """Chronological inner train/validation split of a (possibly non-contiguous) training index.

    The inner split is drawn from ``inner_segment(train, segment)``: by default the
    most recent contiguous run of rows, so that the fitting and validation rows are
    adjacent in time. ``max_rows`` caps fitting and validation rows together in the
    ratio ``1 - val_fraction : val_fraction``; ``max_val_rows``, if given, replaces
    the validation cap so that a longer validation period can be used without
    enlarging the fitting set.

    ``position`` decides where the validation rows lie when a cap applies:

    * ``latest``: the validation set is the most recent rows of the segment (at most
      ``val_fraction`` of it and at most the validation cap) and the fitting set is the
      rows immediately before it. The configuration is therefore judged on the period
      closest to the one it will be used for.
    * ``after_cut`` (versions up to 1.3): the segment is cut at ``1 - val_fraction``;
      the validation set is the earliest capped rows after the cut and the fitting set
      the latest capped rows before it, so rows at the end of the segment can remain
      unused.

    Without caps the two positions coincide.
    """
    train = inner_segment(train, segment)
    n = len(train)
    local_starts = np.flatnonzero(np.isin(train, unit_starts))
    n_fit = int(max_rows * (1 - val_fraction)) if max_rows else None
    n_val = max_val_rows if max_val_rows else (int(max_rows * val_fraction) if max_rows else None)
    if position == "latest":
        n_val_rows = int(round(n * val_fraction))
        if n_val is not None:
            n_val_rows = min(n_val_rows, n_val)
        cut = snap(n - n_val_rows, local_starts, n)
        if cut >= n:                      # snapping consumed the validation set: fall back to the earlier unit start
            i = np.searchsorted(local_starts, n - n_val_rows, side="right") - 1
            cut = int(local_starts[max(i, 0)])
        fit, val = train[:cut], train[cut:]
        if n_fit is not None and len(fit) > n_fit:
            fit = fit[len(fit) - n_fit:]
            s = np.flatnonzero(np.isin(fit, unit_starts))
            fit = fit[s[0]:] if len(s) else fit
        return fit, val
    if position != "after_cut":
        raise ValueError(f"tuning.val_position must be 'latest' or 'after_cut', got '{position}'")
    cut = snap(int(round(n * (1 - val_fraction))), local_starts, n)
    fit, val = train[:cut], train[cut:]
    if n_fit is not None:
        if len(fit) > n_fit:
            fit = fit[len(fit) - n_fit:]
            s = np.flatnonzero(np.isin(fit, unit_starts))
            fit = fit[s[0]:] if len(s) else fit
    if n_val is not None:
        if len(val) > n_val:
            e = np.flatnonzero(np.isin(val, unit_starts))
            end = e[np.searchsorted(e, n_val)] if np.searchsorted(e, n_val) < len(e) else len(val)
            val = val[:end]
    return fit, val


def rolling_origins(n: int, unit_starts: np.ndarray, n_origins: int, test_fraction: float,
                    window: str | float = "expanding") -> list[Fold]:
    """Rolling-origin evaluation windows that tile the most recent part of the data."""
    test_len = int(round(n * test_fraction))
    if test_len * n_origins >= n:
        raise ValueError("rolling.n_origins * rolling.test_fraction must be < 1")
    folds = []
    for j in range(n_origins):
        origin = snap(n - (n_origins - j) * test_len, unit_starts, n)
        end = snap(origin + test_len, unit_starts, n) if j < n_origins - 1 else n
        if window == "expanding":
            start = 0
        else:
            start = snap(max(0, origin - int(round(n * float(window)))), unit_starts, n)
        folds.append(Fold(np.arange(start, origin), np.arange(origin, end)))
    return folds


def describe_folds(folds: list[Fold], times: np.ndarray, label: str) -> pd.DataFrame:
    rows = []
    for i, f in enumerate(folds):
        rows.append({"scheme": label, "fold": i, "n_train": len(f.train), "n_valid": len(f.valid),
                     "train_start": pd.Timestamp(times[f.train].min()), "train_end": pd.Timestamp(times[f.train].max()),
                     "valid_start": pd.Timestamp(times[f.valid].min()), "valid_end": pd.Timestamp(times[f.valid].max())})
    return pd.DataFrame(rows)
