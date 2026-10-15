"""Visitor-level features: an explicitly ablation-only signal.

Retailrocket's ``visitorid`` is a cookie, not a durable account, so the
primary session-kNN method deliberately never conditions on it -- the whole
point of the project is that anonymous, short-lived sessions can be modelled
without any persistent user id. ``visitor_prior_session_count`` exists only
so that claim can be tested empirically (does adding the cookie back in help
at all?) rather than merely asserted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def visitor_prior_session_count(
    session_summary: pd.DataFrame,
    cutoff_start_ts: int | None = None,
) -> pd.Series:
    """Count training sessions a visitor started strictly before a cut-off.

    Args:
        session_summary: One row per session with ``visitorid`` and
            ``start_ts``, e.g. the output of
            :func:`recommender.data.sessionize.session_summary` restricted to
            training sessions.
        cutoff_start_ts: Strict point-in-time boundary on ``start_ts``;
            ``None`` counts every session passed in.

    Returns:
        A series indexed by ``visitorid`` with the count of qualifying prior
        sessions.
    """
    frame = session_summary
    if cutoff_start_ts is not None:
        frame = frame.loc[frame["start_ts"] < cutoff_start_ts]

    counts = frame.groupby("visitorid").size().astype(np.float32)
    counts.name = "visitor_prior_session_count"
    return counts
