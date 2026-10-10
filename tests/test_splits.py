"""Chronological hold-out, out-of-fold blocks, purge gap, inner tuning splits and rolling origins."""
import numpy as np
import pandas as pd
import pytest

from stackxai.splits import holdout, inner_split, rolling_origins, temporal_folds


def grouped_starts(n_groups, size=2):
    return np.arange(0, n_groups * size, size), n_groups * size


def test_holdout_is_chronological_and_respects_groups():
    starts, n = grouped_starts(500)
    train, test = holdout(n, starts, 0.3)
    assert train.max() < test.min()
    assert test.min() in set(starts)
    assert len(train) + len(test) == n


@pytest.mark.parametrize("scheme", ["blocked", "forward"])
def test_folds_are_disjoint_and_cover(scheme):
    starts, n = grouped_starts(600)
    train = np.arange(n)
    folds = temporal_folds(train, starts, 5, scheme)
    covered = np.concatenate([f.valid for f in folds])
    assert len(np.unique(covered)) == len(covered)
    for f in folds:
        assert not np.intersect1d(f.train, f.valid).size
        assert f.valid.min() in set(starts)
        if scheme == "forward":
            assert f.train.max() < f.valid.min()
    expected = n if scheme == "blocked" else n - folds[0].valid.min()
    assert len(covered) == expected


def test_gap_purges_neighbouring_rows():
    n = 1000
    times = pd.date_range("2020-01-01", periods=n, freq="h").to_numpy()
    folds = temporal_folds(np.arange(n), np.arange(n), 4, "blocked", times, gap="12h")
    for f in folds:
        lo, hi = times[f.valid].min(), times[f.valid].max()
        near = (times[f.train] > lo - np.timedelta64(12, "h")) & (times[f.train] < hi + np.timedelta64(12, "h"))
        assert not near.any()


def test_inner_split_is_chronological_subset():
    starts, n = grouped_starts(1000)
    train = np.r_[np.arange(0, 800), np.arange(1200, n)]
    fit, val = inner_split(train, starts, 0.2, max_rows=500)
    assert np.isin(fit, train).all() and np.isin(val, train).all()
    assert fit.max() < val.min()
    assert len(fit) <= 400 + 2 and len(val) <= 100 + 2
    assert fit.min() in set(starts) and val.min() in set(starts)


def test_rolling_origins_never_look_ahead():
    starts, n = grouped_starts(1000)
    for window in ("expanding", 0.4):
        for f in rolling_origins(n, starts, 3, 0.1, window):
            assert f.train.max() < f.valid.min()


def test_inner_split_never_spans_the_held_out_block():
    """Under the blocked scheme an interior fold has a hole; the inner split must stay on one side of it."""
    n, block = 5000, 1000
    starts = np.arange(n)
    folds = temporal_folds(np.arange(n), starts, 5, "blocked")
    for f in folds:
        fit, val = inner_split(f.train, starts, 0.2, max_rows=600)
        both = np.r_[fit, val]
        assert np.all(np.diff(both) == 1), "fit and validation rows must form one contiguous run"
        assert not np.intersect1d(both, f.valid).size
    all_rows_fit, _ = inner_split(folds[3].train, starts, 0.2, max_rows=600, segment="all")
    assert not np.all(np.diff(all_rows_fit) == 1)    # segment="all" crosses the hole


def test_inner_segment_extends_backwards_when_too_short():
    from stackxai.splits import inner_segment
    train = np.r_[np.arange(0, 500), np.arange(600, 700)]
    assert len(inner_segment(train, "contiguous", min_rows=0)) == 100
    assert len(inner_segment(train, "contiguous", min_rows=300)) == 600
    with pytest.raises(ValueError):
        inner_segment(train, "unknown")
