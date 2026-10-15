"""Behavioural tests for the models and the evaluation protocol.

These assert properties that must hold for any correct implementation
(determinism, history exclusion, learning signal) rather than pinning exact
metric values, which would turn routine tuning into a test failure.
"""

from __future__ import annotations

import numpy as np
import pytest

from recommender.evaluation import evaluate, iter_prefixes
from recommender.models import PopularityRecommender, RandomRecommender, SessionKNNRecommender

FAST_KNN = {"k": 30, "sample_size": 200}


@pytest.fixture(scope="module")
def fitted_knn(split):
    return SessionKNNRecommender(**FAST_KNN).fit(split)


def test_unknown_similarity_is_rejected() -> None:
    with pytest.raises(ValueError, match="similarity"):
        SessionKNNRecommender(similarity="euclidean")


def test_fit_rejects_empty_training_data(split) -> None:
    empty = type(split)(
        train=split.train.iloc[0:0],
        validation=split.validation,
        test=split.test,
        n_items=split.n_items,
        train_session_ids=frozenset(),
    )
    with pytest.raises(ValueError):
        SessionKNNRecommender().fit(empty)


def test_neighbours_are_sorted_by_similarity_descending(fitted_knn, split) -> None:
    some_session = next(iter(fitted_knn.session_items))
    query_items = fitted_knn.session_items[some_session]
    query_ts = fitted_knn.session_start_ts[some_session]

    neighbours = fitted_knn.neighbours(query_items, query_ts)
    sims = [sim for _, sim, _ in neighbours]
    assert sims == sorted(sims, reverse=True)


def test_cosine_and_jaccard_agree_on_identical_sessions(split) -> None:
    """sim(S, S) must be 1.0 under both similarity metrics."""
    cosine = SessionKNNRecommender(similarity="cosine")._sim_fn
    jaccard = SessionKNNRecommender(similarity="jaccard")._sim_fn

    s = {1, 2, 3}
    assert cosine(s, s) == pytest.approx(1.0)
    assert jaccard(s, s) == pytest.approx(1.0)


def test_disjoint_sessions_have_zero_similarity() -> None:
    cosine = SessionKNNRecommender(similarity="cosine")._sim_fn
    jaccard = SessionKNNRecommender(similarity="jaccard")._sim_fn

    assert cosine({1, 2}, {3, 4}) == 0.0
    assert jaccard({1, 2}, {3, 4}) == 0.0


def test_recency_decay_discounts_older_neighbours() -> None:
    from recommender.models.session_knn import SIMILARITY_FUNCTIONS

    model = SessionKNNRecommender(k=5, decay_lambda=0.1, similarity="cosine")
    model.n_items = 10
    model.session_items = {"old": {1, 2}, "new": {1, 2}}
    model.session_start_ts = {"old": 0, "new": 1000 * 60 * 60 * 24}  # 24h later
    model.item_to_sessions = {1: ["old", "new"], 2: ["old", "new"]}
    model._sim_fn = SIMILARITY_FUNCTIONS["cosine"]

    query_ts = model.session_start_ts["new"]
    neighbours = {sid: w for sid, _, w in model.neighbours({1, 2}, query_ts)}
    assert neighbours["new"] > neighbours["old"]


@pytest.mark.parametrize(
    "model",
    [
        SessionKNNRecommender(**FAST_KNN),
        PopularityRecommender(),
        RandomRecommender(),
    ],
    ids=["session_knn", "popularity", "random"],
)
def test_recommend_contract(model, split) -> None:
    """Every model returns <=k distinct, in-range items and honours `exclude`."""
    model.fit(split)
    some_session = next(iter(split.train.groupby("session_id")))[1]
    query_items = set(int(i) for i in some_session["item_index"])
    query_ts = int(some_session["session_start_ts"].iloc[0])

    ranked = model.recommend(query_items, query_ts, k=10, exclude=query_items)

    assert len(ranked) <= 10
    assert len(set(ranked.tolist())) == len(ranked), "duplicate items in a single ranking"
    assert all(0 <= int(i) < split.n_items for i in ranked)
    assert not query_items.intersection(int(i) for i in ranked)


def test_session_knn_beats_random_on_recall(fitted_knn, split) -> None:
    knn = evaluate(fitted_knn, split, partition="validation", k=20).recall
    random_model = evaluate(
        RandomRecommender().fit(split), split, partition="validation", k=20
    ).recall
    assert knn > random_model


def test_session_knn_has_higher_coverage_than_popularity(fitted_knn, split) -> None:
    """Catalogue coverage is the business reason to move off a popularity list."""
    knn = evaluate(fitted_knn, split, partition="validation", k=20).coverage
    pop = evaluate(PopularityRecommender().fit(split), split, partition="validation", k=20).coverage
    assert knn > pop


def test_evaluation_excludes_the_query_session_itself(split) -> None:
    """A model that only re-recommends the query items must score zero."""

    class HistoryParrot:
        def recommend(self, query_items, query_start_ts, k=10, exclude=None):  # noqa: ARG002
            seen = sorted(exclude or {0})
            padded = (seen * k)[:k]
            return np.array(padded, dtype=int)

    report = evaluate(HistoryParrot(), split, partition="validation", k=10)
    assert report.recall == 0.0
    assert report.hit_rate == 0.0


def test_unknown_partition_is_rejected(fitted_knn, split) -> None:
    with pytest.raises(ValueError):
        evaluate(fitted_knn, split, partition="train_oops")


def test_cold_session_breakdown_is_reported_separately(fitted_knn, split) -> None:
    report = evaluate(fitted_knn, split, partition="validation", k=20)
    assert report.n_cold > 0
    assert report.n_cold <= report.n_predictions


def test_iter_prefixes_skips_length_one_sessions(split) -> None:
    single = split.validation.groupby("session_id").filter(lambda g: len(g) == 1)
    if single.empty:
        pytest.skip("fixture produced no length-1 validation session this seed")
    produced_sessions = {sid for sid, *_ in iter_prefixes(single)}
    assert produced_sessions == set()


def test_iter_prefixes_yields_length_minus_one_positions(split) -> None:
    multi = split.validation.groupby("session_id").filter(lambda g: len(g) == 3)
    if multi.empty:
        pytest.skip("fixture produced no length-3 validation session this seed")
    one_session_id = multi["session_id"].iloc[0]
    one_session = multi.loc[multi["session_id"] == one_session_id]
    positions = list(iter_prefixes(one_session))
    assert len(positions) == 2


def test_sample_size_truncates_to_the_most_recent_candidates(split) -> None:
    model = SessionKNNRecommender(k=5, sample_size=1).fit(split)
    some_session = next(iter(model.session_items.values()))
    candidates = model._candidate_sessions(some_session)
    assert len(candidates) <= 1
