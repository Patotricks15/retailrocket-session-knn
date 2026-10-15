"""Session-local, point-in-time interaction features.

Both features here are computed per event row from only the *other* events in
the same session that occurred strictly before it -- never from the event's
own timestamp interpreted as "the future", and never from another session.
Because the computation is purely a function of ``(session_id,
session_position)`` ordering, appending further events anywhere in the frame
(even inside the same session, after the row in question) cannot change an
already-computed value; that invariant is what ``tests/test_features_leakage.py``
asserts directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def session_length_so_far(sessionized: pd.DataFrame) -> pd.Series:
    """Number of events already observed in the session before this row.

    This is exactly ``session_position``: the count of strictly-earlier
    events sharing the same ``session_id``.
    """
    series = sessionized["session_position"].astype(np.float32).copy()
    series.name = "session_length_so_far"
    return series


def time_since_session_start(
    sessionized: pd.DataFrame, unit_seconds: bool = True
) -> pd.Series:
    """Elapsed time between the session start and the last *observed* event.

    At session position ``p`` this reads the timestamp of the event at
    position ``p - 1`` (the most recent event the model has actually seen),
    not the timestamp of the row being scored itself, which is the quantity
    the model is trying to predict alongside the next item.

    Args:
        sessionized: Frame with ``session_id``, ``session_position``,
            ``timestamp`` and ``session_start_ts``.
        unit_seconds: Convert from the raw millisecond timestamps to seconds.

    Returns:
        A ``float32`` series aligned to ``sessionized``'s row order; position
        0 of every session has no prior event and is filled with ``0.0``.
    """
    frame = sessionized.sort_values(["session_id", "session_position"], kind="mergesort")
    previous_ts = frame.groupby("session_id")["timestamp"].shift(1)
    elapsed = (previous_ts - frame["session_start_ts"]).fillna(0.0)

    if unit_seconds:
        elapsed = elapsed / 1000.0

    series = elapsed.astype(np.float32).reindex(sessionized.index)
    series.name = "time_since_session_start"
    return series
