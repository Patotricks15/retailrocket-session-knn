"""Strictly temporal, session-level train/validation/test splitting.

Row-level (event-level) splits are unsafe for session models: cutting a
session in half and putting the first events in train and the rest in test
lets the model "see" part of a test session during feature fitting. The
split implemented here instead partitions whole sessions by their start time,
so a session belongs entirely to one partition.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from recommender.data.sessionize import session_summary


@dataclass(frozen=True)
class SessionSplit:
    """A leakage-safe, session-level partition of the sessionized events.

    ``train`` holds every event belonging to a session that started strictly
    before the validation cut-off time, so any aggregate fitted on ``train``
    is automatically point-in-time correct with respect to both evaluation
    partitions. ``train_session_ids`` is the candidate neighbour pool for
    session-kNN: neighbours are only ever drawn from training sessions, never
    from other validation/test sessions, which would otherwise leak across
    "visitors of the future".

    Attributes:
        train: Events whose session falls in the train partition.
        validation: Events whose session falls in the validation partition.
        test: Events whose session falls in the test partition.
        n_items: Size of the item catalogue (for score vector allocation).
        train_session_ids: Session ids available as k-NN neighbours.
    """

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    n_items: int
    train_session_ids: frozenset[str]

    def session_counts(self) -> dict[str, int]:
        return {
            "train": self.train["session_id"].nunique(),
            "validation": self.validation["session_id"].nunique(),
            "test": self.test["session_id"].nunique(),
        }


def temporal_session_split(
    sessionized: pd.DataFrame,
    n_items: int,
    train_frac: float = 0.70,
    validation_frac: float = 0.15,
    min_session_length: int = 2,
) -> SessionSplit:
    """Partition sessions into train/validation/test by start-time quantiles.

    Sessions of length 1 carry no "next item" to predict and are kept only in
    train, where they still contribute to item-popularity and visitor-history
    features; they are never placed in validation/test since the next-item
    protocol cannot evaluate them (see the "cold session" discussion in
    ``paper/sections/05-evaluation.tex``).

    Args:
        sessionized: Output of :func:`recommender.data.sessionize.sessionize`.
        n_items: Size of the item catalogue.
        train_frac: Fraction of sessions (by start time) assigned to train.
        validation_frac: Fraction assigned to validation; the remainder
            becomes test.
        min_session_length: Minimum events a session needs to be eligible for
            validation/test evaluation.

    Returns:
        A :class:`SessionSplit`.

    Raises:
        ValueError: If the fractions are invalid or no session qualifies for
            evaluation.
    """
    if not 0 < train_frac < 1 or not 0 < validation_frac < 1:
        raise ValueError("train_frac and validation_frac must be in (0, 1).")
    if train_frac + validation_frac >= 1:
        raise ValueError("train_frac + validation_frac must be < 1 to leave a test split.")

    summary = session_summary(sessionized)
    n_sessions = summary.shape[0]

    train_cut = int(np.floor(n_sessions * train_frac))
    val_cut = int(np.floor(n_sessions * (train_frac + validation_frac)))

    # summary is already sorted by start_ts; splitting by position is
    # equivalent to splitting by a start-time threshold and avoids ties.
    train_ids = set(summary["session_id"].iloc[:train_cut])
    val_ids = set(summary["session_id"].iloc[train_cut:val_cut])
    test_ids = set(summary["session_id"].iloc[val_cut:])

    eligible_lengths = summary.set_index("session_id")["length"]
    val_ids = {sid for sid in val_ids if eligible_lengths[sid] >= min_session_length}
    test_ids = {sid for sid in test_ids if eligible_lengths[sid] >= min_session_length}

    if not test_ids:
        raise ValueError(
            "No session qualifies for the test partition; lower min_session_length "
            "or increase the dataset size."
        )

    train = sessionized.loc[sessionized["session_id"].isin(train_ids)].reset_index(drop=True)
    validation = sessionized.loc[sessionized["session_id"].isin(val_ids)].reset_index(drop=True)
    test = sessionized.loc[sessionized["session_id"].isin(test_ids)].reset_index(drop=True)

    return SessionSplit(
        train=train,
        validation=validation,
        test=test,
        n_items=n_items,
        train_session_ids=frozenset(train_ids),
    )
