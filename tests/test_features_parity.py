"""Training/serving parity tests.

The failure mode these guard against is skew: the offline evaluation scores
one ranking while production serves a different one. For session-kNN the
serving path recomputes the ranking at request time (there is no per-session
precompute target), so parity here means "the Lambda handler's score for a
given partial session is byte-for-byte what `model.recommend` produces
offline" -- plus the precomputed popularity fallback used for cold/empty
sessions.
"""

from __future__ import annotations

import json
from decimal import Decimal

import pytest

from recommender.models import PopularityRecommender, SessionKNNRecommender
from recommender.serving import build_popularity_fallback_payload, parse_request
from recommender.serving.handler import _DynamoEncoder, _model_cache, lambda_handler


@pytest.fixture
def model(split):
    return SessionKNNRecommender(k=50, sample_size=300).fit(split)


@pytest.fixture
def popularity_payload(split, event_data):
    pop = PopularityRecommender().fit(split)
    return build_popularity_fallback_payload(pop.item_scores, event_data.item_ids, k=10)


def test_fallback_payload_covers_k_items(popularity_payload) -> None:
    assert len(popularity_payload["recommendations"]) == 10


def test_fallback_payload_ranks_are_contiguous(popularity_payload) -> None:
    ranks = [r["rank"] for r in popularity_payload["recommendations"]]
    assert ranks == list(range(1, 11))


def test_fallback_payload_is_json_serialisable(popularity_payload) -> None:
    json.dumps(popularity_payload)


def test_fallback_rejects_misaligned_arrays(split, event_data) -> None:
    pop = PopularityRecommender().fit(split)
    with pytest.raises(ValueError, match="item_ids"):
        build_popularity_fallback_payload(pop.item_scores, event_data.item_ids[:-1])


def test_fallback_rejects_non_positive_k(split, event_data) -> None:
    pop = PopularityRecommender().fit(split)
    with pytest.raises(ValueError, match="k must be positive"):
        build_popularity_fallback_payload(pop.item_scores, event_data.item_ids, k=0)


def test_handler_scores_a_known_session_identically_to_offline_recommend(
    model, event_data, split
) -> None:
    """The Lambda handler's ranking must equal the offline `recommend` ranking."""
    _model_cache.clear()
    _model_cache["model"] = model
    _model_cache["item_ids"] = event_data.item_ids
    _model_cache["item_index"] = {
        str(int(iid)): idx for idx, iid in enumerate(event_data.item_ids)
    }

    first_session = split.test.sort_values("session_position").iloc[0]["session_id"]
    rows = split.test.loc[split.test["session_id"] == first_session].sort_values(
        "session_position"
    )
    seen_items = set(int(i) for i in rows["item_index"].iloc[:1])
    itemids = [str(int(event_data.item_ids[i])) for i in seen_items]

    offline_ranked = model.recommend(seen_items, int(rows["session_start_ts"].iloc[0]), k=10,
                                      exclude=seen_items)
    offline_ids = [str(int(event_data.item_ids[int(i)])) for i in offline_ranked]

    response = lambda_handler(
        {"items": itemids, "k": 10, "query_start_ts": int(rows["session_start_ts"].iloc[0])}
    )
    served_ids = [r["itemid"] for r in json.loads(response["body"])["recommendations"]]

    assert served_ids == offline_ids, "serving skew between the handler and offline recommend"
    _model_cache.clear()


def test_handler_falls_back_when_every_item_is_unknown(model, event_data) -> None:
    _model_cache.clear()
    _model_cache["model"] = model
    _model_cache["item_ids"] = event_data.item_ids
    _model_cache["item_index"] = {
        str(int(iid)): idx for idx, iid in enumerate(event_data.item_ids)
    }

    class _FakeTable:
        def get_item(self, Key):  # noqa: N803 - matches boto3's signature
            return {
                "Item": {
                    "session_key": "GLOBAL",
                    "recommendations": [{"rank": 1, "itemid": "999999"}],
                }
            }

    _model_cache["table"] = _FakeTable()
    response = lambda_handler({"items": ["not-a-real-item"], "k": 5})
    body = json.loads(response["body"])

    assert response["statusCode"] == 200
    assert body["model"] == "popularity-fallback"
    _model_cache.clear()


# --- Request validation (no AWS required) -----------------------------------


def test_empty_items_list_is_accepted() -> None:
    items, k, ts = parse_request({"items": [], "k": 5})
    assert items == []
    assert k == 5
    assert ts is None


def test_valid_request_is_parsed() -> None:
    items, k, ts = parse_request({"items": ["7", "11"], "k": 5})
    assert items == ["7", "11"]
    assert k == 5
    assert ts is None


def test_request_body_as_json_string_is_parsed() -> None:
    items, k, _ts = parse_request({"body": json.dumps({"items": ["3"], "k": 2})})
    assert items == ["3"]
    assert k == 2


def test_explicit_query_start_ts_is_parsed() -> None:
    items, k, ts = parse_request({"items": ["3"], "k": 2, "query_start_ts": 123456})
    assert items == ["3"]
    assert k == 2
    assert ts == 123456


@pytest.mark.parametrize("bad_k", [0, -1, 101, "abc"])
def test_invalid_k_is_rejected(bad_k) -> None:
    with pytest.raises(ValueError, match="k"):
        parse_request({"items": [], "k": bad_k})


def test_non_list_items_is_rejected() -> None:
    with pytest.raises(ValueError, match="items"):
        parse_request({"items": "not-a-list"})


def test_handler_returns_400_on_bad_request() -> None:
    response = lambda_handler({"items": "oops"})
    assert response["statusCode"] == 400
    assert "items" in json.loads(response["body"])["error"]


def test_response_serialises_dynamodb_decimals() -> None:
    body = {
        "k": Decimal("3"),
        "recommendations": [{"rank": Decimal("1"), "itemid": "275"}],
        "score": Decimal("0.5"),
    }
    decoded = json.loads(json.dumps(body, cls=_DynamoEncoder))

    assert decoded["k"] == 3
    assert isinstance(decoded["k"], int)
    assert decoded["recommendations"][0]["rank"] == 1
    assert decoded["score"] == 0.5


def test_encoder_still_rejects_genuinely_unserialisable_types() -> None:
    with pytest.raises(TypeError):
        json.dumps({"bad": object()}, cls=_DynamoEncoder)
