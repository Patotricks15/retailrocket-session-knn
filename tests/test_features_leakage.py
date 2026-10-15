"""Leakage tests.

These are the most important tests in the repository. A recommender that
leaks future information reports excellent offline metrics and then fails in
production, so each leakage class gets an explicit, failing-by-default check.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from recommender.features import item_features


def test_future_event_does_not_change_past_popularity(split) -> None:
    """Injecting an event after the cut-off must not alter any past value.

    This is the direct operationalization of the point-in-time rule: the
    feature computed at cut-off C must be identical whether or not events at
    positions >= C exist in the frame.
    """
    train = split.train
    cutoff = int(train["event_order"].max())

    before = item_features.item_popularity_count(train, n_items=split.n_items, cutoff_order=cutoff)

    future_row = train.iloc[[0]].copy()
    future_row["event_order"] = cutoff + 50
    polluted = pd.concat([train, future_row], ignore_index=True)

    after = item_features.item_popularity_count(
        polluted, n_items=split.n_items, cutoff_order=cutoff
    )

    np.testing.assert_array_equal(before.to_numpy(), after.to_numpy())


def test_cutoff_boundary_is_strictly_exclusive(split) -> None:
    """An event exactly at the cut-off belongs to the future, not the past."""
    train = split.train
    item = int(train["item_index"].iloc[0])
    item_rows = train.loc[train["item_index"] == item]
    cutoff = int(item_rows["event_order"].max())

    counts = item_features.item_popularity_count(train, n_items=split.n_items, cutoff_order=cutoff)
    expected = int((item_rows["event_order"] < cutoff).sum())

    assert int(counts.loc[item]) == expected


def test_popularity_is_fitted_on_training_partition_only(split, event_data) -> None:
    """Popularity computed on train must differ from popularity on all data.

    If they were identical, the training statistic would already encode the
    held-out sessions and the evaluation would be optimistic.
    """
    train_pop = item_features.item_popularity_count(split.train, n_items=split.n_items)
    full_pop = item_features.item_popularity_count(event_data.frame, n_items=split.n_items)

    assert not np.array_equal(train_pop.to_numpy(), full_pop.to_numpy()), (
        "train-only popularity matches full-data popularity; the fit scope is not enforced"
    )


def test_session_length_so_far_is_unaffected_by_future_events_in_other_sessions(
    feature_store, split
) -> None:
    """A session-local feature must not depend on unrelated sessions at all."""
    frame = split.validation
    before = feature_store.augment_events(frame)["session_length_so_far"].to_numpy()

    polluted = pd.concat([frame, split.test], ignore_index=True)
    after = feature_store.augment_events(polluted)["session_length_so_far"].to_numpy()[
        : len(before)
    ]

    np.testing.assert_array_equal(before, after)


def test_time_since_session_start_ignores_the_event_being_scored(feature_store) -> None:
    """The elapsed-time feature must read the *previous* event, not the current one.

    Position 0 of every session has no prior event, so it must be filled with
    zero rather than leaking the (unknown-at-prediction-time) current
    timestamp.
    """
    frame = pd.DataFrame(
        {
            "session_id": ["a", "a", "a"],
            "session_position": [0, 1, 2],
            "timestamp": [1_000, 5_000, 9_000],
            "session_start_ts": [1_000, 1_000, 1_000],
        }
    )
    augmented = feature_store.augment_events(frame)
    elapsed = augmented["time_since_session_start"].to_numpy()

    assert elapsed[0] == 0.0
    assert elapsed[1] == pytest.approx(0.0)  # previous event (pos 0) was at session start
    assert elapsed[2] == pytest.approx(4.0)  # previous event (pos 1) was 4s after start


def test_train_validation_test_sessions_are_disjoint(split) -> None:
    train = set(split.train["session_id"])
    validation = set(split.validation["session_id"])
    test = set(split.test["session_id"])

    assert not (train & validation)
    assert not (train & test)
    assert not (validation & test)


def test_test_sessions_start_after_every_train_session(split) -> None:
    """The temporal split must not interleave train and test session starts."""
    train_max_start = split.train["session_start_ts"].max()
    test_min_start = split.test["session_start_ts"].min()
    assert test_min_start >= train_max_start


def test_evaluation_neighbours_are_drawn_only_from_train(split) -> None:
    """Session-kNN's candidate pool must never include validation/test sessions."""
    from recommender.models import SessionKNNRecommender

    model = SessionKNNRecommender(k=10).fit(split)
    indexed = set(model.session_items)

    assert indexed == split.train_session_ids
    assert not (indexed & set(split.validation["session_id"]))
    assert not (indexed & set(split.test["session_id"]))


def test_shuffled_item_assignment_destroys_ranking_quality(split) -> None:
    """A model scored against shuffled item identities must not beat chance.

    This catches leakage that survives the structural checks: if performance
    stayed high after destroying the item identity within each session, a
    feature or an index would be carrying the answer by accident.
    """
    from recommender.evaluation import evaluate
    from recommender.models import SessionKNNRecommender

    rng = np.random.default_rng(0)
    shuffled = split.train.copy()
    shuffled["item_index"] = rng.permutation(shuffled["item_index"].to_numpy())

    broken_split = type(split)(
        train=shuffled,
        validation=split.validation,
        test=split.test,
        n_items=split.n_items,
        train_session_ids=split.train_session_ids,
    )

    model = SessionKNNRecommender(k=20, sample_size=200).fit(broken_split)
    report = evaluate(model, broken_split, partition="validation", k=20)
    random_floor = 20 / split.n_items

    assert report.recall < max(0.05, random_floor * 6), (
        f"recall@20 stayed at {report.recall:.4f} after shuffling item ids; a feature is leaking"
    )
