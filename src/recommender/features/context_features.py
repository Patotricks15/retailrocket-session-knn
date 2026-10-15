"""Context features: deterministic functions of the event's own timestamp.

Unlike the interaction features, these are not aggregates over history at
all, so there is no cut-off to enforce: the wall-clock time of the request
being scored is known by definition at prediction time. They are declared
``point_in_time: false`` in ``conf/features.yaml`` for exactly this reason.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def hour_of_day(sessionized: pd.DataFrame) -> pd.Series:
    """Hour (0-23) of the event being scored, in UTC."""
    hours = pd.to_datetime(sessionized["timestamp"], unit="ms", utc=True).dt.hour
    series = hours.astype(np.float32)
    series.name = "hour_of_day"
    return series


def is_weekend(sessionized: pd.DataFrame) -> pd.Series:
    """Whether the event being scored falls on a Saturday or Sunday."""
    weekday = pd.to_datetime(sessionized["timestamp"], unit="ms", utc=True).dt.weekday
    series = (weekday >= 5).astype(np.float32)
    series.name = "is_weekend"
    return series
