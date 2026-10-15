"""Sessionization by time-gap segmentation.

Retailrocket carries no explicit session id: a visitor's event stream must be
cut into sessions ourselves. The standard, interpretable rule -- used by the
session-kNN literature this project follows -- is a fixed inactivity
threshold: a gap of more than ``gap_seconds`` (30 minutes by default) between
two consecutive events from the same visitor starts a new session.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_GAP_SECONDS = 30 * 60


def sessionize(events: pd.DataFrame, gap_seconds: int = DEFAULT_GAP_SECONDS) -> pd.DataFrame:
    """Segment a visitor's event stream into sessions by inactivity gap.

    Args:
        events: Frame with at least ``visitorid``, ``timestamp`` (milliseconds
            since epoch) and ``item_index``. Any additional columns (e.g.
            ``event_order``, ``event``) are preserved.
        gap_seconds: Inactivity threshold; a gap strictly greater than this
            starts a new session for that visitor.

    Returns:
        A copy of ``events`` sorted by ``(visitorid, timestamp)`` with three
        new columns:

        ``session_id``
            A string unique across the whole frame, ``"{visitorid}_{seq}"``.
        ``session_position``
            0-indexed position of the event within its session.
        ``session_start_ts``
            Timestamp of the first event in the session, broadcast to every
            row of that session -- used by the temporal split and by the
            recency-weighting of neighbour sessions.

    Raises:
        ValueError: If ``gap_seconds`` is not positive.
    """
    if gap_seconds <= 0:
        raise ValueError(f"gap_seconds must be positive, got {gap_seconds}.")

    frame = events.sort_values(["visitorid", "timestamp"], kind="mergesort").reset_index(drop=True)

    gap_ms = gap_seconds * 1000
    same_visitor = frame["visitorid"] == frame["visitorid"].shift(1)
    delta = frame["timestamp"] - frame["timestamp"].shift(1)
    is_new_session = ~(same_visitor & (delta <= gap_ms))

    session_seq = is_new_session.groupby(frame["visitorid"]).cumsum().astype(np.int64)
    frame["session_id"] = (
        frame["visitorid"].astype(str) + "_" + session_seq.astype(str)
    )
    frame["session_position"] = frame.groupby("session_id").cumcount().astype(np.int32)
    frame["session_start_ts"] = frame.groupby("session_id")["timestamp"].transform("min")

    return frame


def session_summary(sessionized: pd.DataFrame) -> pd.DataFrame:
    """One row per session: length, start/end time, visitor.

    Used by the temporal splitter and by reporting (coverage, cold-session
    breakdown) so those consumers never have to re-derive session boundaries.
    """
    grouped = sessionized.groupby("session_id", sort=False)
    summary = grouped.agg(
        visitorid=("visitorid", "first"),
        length=("session_position", "size"),
        start_ts=("timestamp", "min"),
        end_ts=("timestamp", "max"),
    ).reset_index()
    return summary.sort_values("start_ts", kind="mergesort").reset_index(drop=True)
