"""Build the cold-start fallback payload published to DynamoDB.

Session-kNN has no stable per-entity key to precompute against: a
``visitorid`` is an unreliable cookie and a session does not exist until it
has started, so there is no "user" to materialize a top-N list for ahead of
time. The only recommendation surface that *is* safe to precompute is a
single, session-independent popularity fallback, used by the inference
Lambda when a request carries an empty or unrecognised session. Every other
request is scored on-the-fly against the session-kNN index -- the
architecturally honest choice for an anonymous, request-time method.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def build_popularity_fallback_payload(
    item_scores: np.ndarray,
    item_ids: np.ndarray,
    k: int = 20,
    model_name: str = "popularity-fallback",
) -> dict[str, Any]:
    """Materialize the single DynamoDB item used for cold/empty sessions.

    Args:
        item_scores: Per-item popularity score, aligned to ``item_ids``.
        item_ids: Original ``itemid`` values aligned to internal indices.
        k: List length to store.
        model_name: Stored alongside the list for attribution.

    Returns:
        A dict ready for ``table.put_item``, keyed by the constant
        partition key ``"GLOBAL"`` (there is no per-session row to key on).

    Raises:
        ValueError: If ``k`` is not positive or the arrays are misaligned.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}.")
    if item_scores.shape[0] != item_ids.shape[0]:
        raise ValueError(
            f"item_scores has {item_scores.shape[0]} entries but item_ids has "
            f"{item_ids.shape[0]}."
        )

    k = min(k, item_scores.shape[0])
    top = np.argpartition(-item_scores, k - 1)[:k]
    ordered = top[np.argsort(-item_scores[top])]

    return {
        "session_key": "GLOBAL",
        "model": model_name,
        "recommendations": [
            {"rank": rank, "itemid": str(int(item_ids[int(idx)]))}
            for rank, idx in enumerate(ordered, start=1)
        ],
    }
