"""AWS Lambda entry point: on-the-fly session-kNN scoring, with a precomputed
popularity fallback for empty or fully cold-start sessions.

Deliberately dependency-free beyond boto3 (which the Lambda runtime already
provides) and the pickled session-kNN artifact pulled from S3 at cold start,
so the deployment package stays a single file.
"""

from __future__ import annotations

import json
import os
import pickle
import time
from decimal import Decimal
from typing import Any

try:  # pragma: no cover - boto3 is provided by the Lambda runtime.
    import boto3
except ImportError:  # pragma: no cover
    boto3 = None

DEFAULT_K = int(os.environ.get("DEFAULT_K", "10"))
MAX_K = 100
FALLBACK_TABLE = os.environ.get("FALLBACK_TABLE", "retailrocket-knn-dev-fallback")
MODEL_BUCKET = os.environ.get("ARTIFACTS_BUCKET", "")
MODEL_KEY = os.environ.get("MODEL_KEY", "models/session_knn.pkl")

_model_cache: dict[str, Any] = {}


class _DynamoEncoder(json.JSONEncoder):
    """Serialize the ``Decimal`` values DynamoDB returns for every number.

    boto3's resource API deserializes DynamoDB's ``N`` type to ``Decimal``,
    which ``json.dumps`` rejects outright.
    """

    def default(self, o: Any) -> Any:
        if isinstance(o, Decimal):
            return int(o) if o == o.to_integral_value() else float(o)
        return super().default(o)


def _get_table():
    """Create the DynamoDB resource once per container, not per request."""
    if "table" not in _model_cache:
        if boto3 is None:
            raise RuntimeError("boto3 is unavailable; cannot reach DynamoDB.")
        endpoint = os.environ.get("AWS_ENDPOINT_URL")
        resource = (
            boto3.resource("dynamodb", endpoint_url=endpoint)
            if endpoint
            else boto3.resource("dynamodb")
        )
        _model_cache["table"] = resource.Table(FALLBACK_TABLE)
    return _model_cache["table"]


def _get_model():
    """Load (and memoize) the session-kNN artifact for the life of the container."""
    if "model" in _model_cache:
        return _model_cache["model"], _model_cache["item_index"]
    if boto3 is None:
        raise RuntimeError("boto3 is unavailable; cannot reach S3.")

    endpoint = os.environ.get("AWS_ENDPOINT_URL")
    client = boto3.client("s3", endpoint_url=endpoint) if endpoint else boto3.client("s3")
    response = client.get_object(Bucket=MODEL_BUCKET, Key=MODEL_KEY)
    payload = pickle.loads(response["Body"].read())

    model = payload["model"]
    item_ids = payload["item_ids"]
    item_index = {str(int(iid)): idx for idx, iid in enumerate(item_ids)}

    _model_cache["model"] = model
    _model_cache["item_index"] = item_index
    _model_cache["item_ids"] = item_ids
    return model, item_index


def _response(status: int, body: dict[str, Any]) -> dict[str, Any]:
    return {
        "statusCode": status,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body, cls=_DynamoEncoder),
    }


def parse_request(event: dict[str, Any]) -> tuple[list[str], int, int | None]:
    """Extract and validate ``items``, ``k`` and ``query_start_ts``.

    Split out from the handler so the validation rules are unit-testable
    without any AWS dependency.

    Args:
        event: API Gateway proxy event (JSON body), or a bare dict for direct
            invocation.

    Returns:
        ``(itemids, k, query_start_ts)``, where ``itemids`` is the ordered
        list of items viewed so far in the current (partial) session (it may
        be empty) and ``query_start_ts`` is the optional, explicit session
        start time in epoch milliseconds used to weight recency; when absent
        the handler falls back to the current wall-clock time.

    Raises:
        ValueError: If ``k`` or ``query_start_ts`` are malformed, or
            ``items`` is not a list.
    """
    body = event.get("body")
    if isinstance(body, str):
        try:
            payload = json.loads(body) if body else {}
        except json.JSONDecodeError as exc:
            raise ValueError("Request body is not valid JSON.") from exc
    else:
        payload = event

    items = payload.get("items", [])
    if not isinstance(items, list):
        raise ValueError("Field 'items' must be a list of item ids.")

    raw_k = payload.get("k", DEFAULT_K)
    try:
        k = int(raw_k)
    except (TypeError, ValueError):
        raise ValueError(f"Parameter 'k' must be an integer, got {raw_k!r}.") from None
    if not 1 <= k <= MAX_K:
        raise ValueError(f"Parameter 'k' must be between 1 and {MAX_K}, got {k}.")

    query_start_ts = payload.get("query_start_ts")
    if query_start_ts is not None:
        try:
            query_start_ts = int(query_start_ts)
        except (TypeError, ValueError):
            raise ValueError(
                f"Parameter 'query_start_ts' must be an integer, got {query_start_ts!r}."
            ) from None

    return [str(i) for i in items], k, query_start_ts


def lambda_handler(event: dict[str, Any], context: Any = None) -> dict[str, Any]:
    """Score the posted partial session, or serve the popularity fallback."""
    try:
        itemids, k, query_start_ts = parse_request(event)
    except ValueError as exc:
        return _response(400, {"error": str(exc)})

    try:
        model, item_index = _get_model()
    except Exception as exc:  # noqa: BLE001 - surface as 503, never leak a stack trace.
        return _response(503, {"error": "model artifact unavailable", "detail": str(exc)})

    known_indices = {item_index[i] for i in itemids if i in item_index}

    if not known_indices:
        try:
            item = _get_table().get_item(Key={"session_key": "GLOBAL"}).get("Item")
        except Exception as exc:  # noqa: BLE001
            return _response(
                503, {"error": "fallback store unavailable", "detail": str(exc)}
            )
        recommendations = (item or {}).get("recommendations", [])[:k]
        return _response(
            200,
            {"model": "popularity-fallback", "k": k, "recommendations": recommendations},
        )

    if query_start_ts is None:
        query_start_ts = int(time.time() * 1000)
    ranked = model.recommend(known_indices, query_start_ts, k=k, exclude=known_indices)
    item_ids = _model_cache["item_ids"]
    recommendations = [
        {"rank": rank, "itemid": str(int(item_ids[int(i)]))}
        for rank, i in enumerate(ranked, start=1)
    ]
    return _response(200, {"model": "session-knn", "k": k, "recommendations": recommendations})
