"""Unit tests for time-gap sessionization and the temporal session split."""

from __future__ import annotations

import pandas as pd

from recommender.data.sessionize import session_summary, sessionize
from recommender.data.splitting import temporal_session_split


def _events(rows: list[tuple[int, int, int]]) -> pd.DataFrame:
    """Build a minimal events frame: (visitorid, timestamp_ms, item_index)."""
    return pd.DataFrame(
        [
            {"visitorid": v, "timestamp": t, "item_index": i, "event_order": idx}
            for idx, (v, t, i) in enumerate(rows)
        ]
    )


def test_gap_above_threshold_starts_a_new_session() -> None:
    thirty_one_min = 31 * 60 * 1000
    frame = _events([(1, 0, 10), (1, thirty_one_min, 11)])
    sessionized = sessionize(frame, gap_seconds=30 * 60)

    assert sessionized["session_id"].nunique() == 2


def test_gap_below_threshold_stays_in_one_session() -> None:
    twenty_nine_min = 29 * 60 * 1000
    frame = _events([(1, 0, 10), (1, twenty_nine_min, 11)])
    sessionized = sessionize(frame, gap_seconds=30 * 60)

    assert sessionized["session_id"].nunique() == 1


def test_different_visitors_never_share_a_session() -> None:
    frame = _events([(1, 0, 10), (2, 100, 11)])
    sessionized = sessionize(frame, gap_seconds=30 * 60)

    assert sessionized["session_id"].nunique() == 2


def test_session_position_is_zero_indexed_and_contiguous() -> None:
    frame = _events([(1, 0, 10), (1, 1_000, 11), (1, 2_000, 12)])
    sessionized = sessionize(frame, gap_seconds=30 * 60)

    assert sessionized["session_position"].tolist() == [0, 1, 2]


def test_session_start_ts_is_broadcast_across_the_session() -> None:
    frame = _events([(1, 500, 10), (1, 1_500, 11)])
    sessionized = sessionize(frame, gap_seconds=30 * 60)

    assert sessionized["session_start_ts"].unique().tolist() == [500]


def test_invalid_gap_seconds_is_rejected() -> None:
    import pytest

    frame = _events([(1, 0, 10)])
    with pytest.raises(ValueError):
        sessionize(frame, gap_seconds=0)


def test_session_summary_is_sorted_by_start_time() -> None:
    frame = _events([(2, 5_000, 10), (1, 0, 11), (1, 50_000 * 1000, 12)])
    sessionized = sessionize(frame, gap_seconds=30 * 60)
    summary = session_summary(sessionized)

    assert summary["start_ts"].is_monotonic_increasing


def test_temporal_split_keeps_sessions_whole(split) -> None:
    """No session_id may appear with events spread across two partitions."""
    train_ids = set(split.train["session_id"])
    val_ids = set(split.validation["session_id"])
    test_ids = set(split.test["session_id"])
    assert not (train_ids & val_ids) and not (train_ids & test_ids) and not (val_ids & test_ids)


def test_temporal_split_excludes_short_sessions_from_eval(split) -> None:
    val_lengths = split.validation.groupby("session_id").size()
    test_lengths = split.test.groupby("session_id").size()

    assert (val_lengths >= 2).all()
    assert (test_lengths >= 2).all()


def test_temporal_split_rejects_invalid_fractions(split, event_data, sessionized) -> None:
    import pytest

    with pytest.raises(ValueError):
        temporal_session_split(sessionized, event_data.n_items, train_frac=0.9, validation_frac=0.2)
