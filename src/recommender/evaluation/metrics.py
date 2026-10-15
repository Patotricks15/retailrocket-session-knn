"""Next-item-in-session evaluation: Recall@K, MRR, coverage, cold-session split.

For every session in the evaluated partition, the protocol walks every
prefix length ``p = 1 .. len(session) - 1``: the first ``p`` items become the
query session, the item at position ``p`` is the held-out target, and the
items already seen in the session are excluded from the candidate set (a
model should not "recommend" what the visitor just viewed). Neighbour
sessions are only ever drawn from the training partition by construction of
:class:`recommender.data.splitting.SessionSplit`, so no partial session in
validation/test is ever used as a neighbour for another evaluated session.

The ``p == 1`` prefixes -- exactly one observed item -- are also reported
separately as the "cold session" regime: the hardest, least-informed queries,
and the ones short, anonymous Retailrocket sessions produce most often.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd


class SessionRecommender(Protocol):
    """Structural interface shared by session-kNN and the baselines."""

    def recommend(
        self,
        query_items: set[int],
        query_start_ts: int,
        k: int = 10,
        exclude: set[int] | None = None,
    ) -> np.ndarray: ...


@dataclass(frozen=True)
class EvaluationReport:
    """Metrics for one model on one partition, overall and for cold sessions."""

    model: str
    k: int
    recall: float
    mrr: float
    hit_rate: float
    coverage: float
    n_predictions: int
    recall_cold: float
    mrr_cold: float
    n_cold: int

    def as_row(self) -> dict[str, float | str | int]:
        return {
            "model": self.model,
            "k": self.k,
            "recall": round(self.recall, 4),
            "mrr": round(self.mrr, 4),
            "hit_rate": round(self.hit_rate, 4),
            "coverage": round(self.coverage, 4),
            "n_predictions": self.n_predictions,
            "recall_cold": round(self.recall_cold, 4),
            "mrr_cold": round(self.mrr_cold, 4),
            "n_cold": self.n_cold,
        }


def iter_prefixes(frame: pd.DataFrame):
    """Yield ``(session_id, prefix_items, query_start_ts, target_item, prefix_len)``.

    Args:
        frame: A sessionized partition (validation or test), one row per
            event, carrying ``session_id``, ``session_position``,
            ``item_index`` and ``session_start_ts``.

    Yields:
        One tuple per evaluable position across every session in ``frame``.
        A session of length ``L`` yields ``L - 1`` positions (``p = 1`` is
        the cold-session case); length-1 sessions yield nothing, since they
        have no next item to predict.
    """
    ordered = frame.sort_values(["session_id", "session_position"], kind="mergesort")
    for session_id, group in ordered.groupby("session_id", sort=False):
        items = group["item_index"].to_numpy()
        start_ts = int(group["session_start_ts"].iloc[0])
        for p in range(1, len(items)):
            yield session_id, items[:p], start_ts, int(items[p]), p


def evaluate(
    model: SessionRecommender,
    split,
    partition: str = "test",
    k: int = 20,
    model_name: str = "model",
) -> EvaluationReport:
    """Compute the next-item ranking metrics for a model on a held-out partition.

    Args:
        model: Any object exposing the :class:`SessionRecommender` interface.
        split: A :class:`recommender.data.splitting.SessionSplit`.
        partition: ``"validation"`` or ``"test"``.
        k: Cut-off.
        model_name: Label used in the results table.

    Returns:
        A populated :class:`EvaluationReport`.

    Raises:
        ValueError: If the partition name is unknown, empty, or yields no
            evaluable prefix.
    """
    if partition not in {"validation", "test"}:
        raise ValueError(f"Unknown partition '{partition}'; expected 'validation' or 'test'.")
    frame: pd.DataFrame = getattr(split, partition)
    if frame.empty:
        raise ValueError(f"Partition '{partition}' is empty.")

    recall_total = 0.0
    rr_total = 0.0
    hits = 0
    n = 0
    recall_cold = 0.0
    rr_cold = 0.0
    n_cold = 0
    recommended_items: set[int] = set()

    for _session_id, prefix, start_ts, target, prefix_len in iter_prefixes(frame):
        query_items = set(int(i) for i in prefix)
        ranked = model.recommend(query_items, start_ts, k=k, exclude=query_items)
        recommended_items.update(int(i) for i in ranked)

        positions = np.flatnonzero(ranked == target)
        hit = positions.size > 0
        rr = 1.0 / (int(positions[0]) + 1) if hit else 0.0

        n += 1
        recall_total += float(hit)
        rr_total += rr
        hits += int(hit)

        if prefix_len == 1:
            n_cold += 1
            recall_cold += float(hit)
            rr_cold += rr

    if n == 0:
        raise ValueError(f"Partition '{partition}' produced no evaluable session prefix.")

    return EvaluationReport(
        model=model_name,
        k=k,
        recall=recall_total / n,
        mrr=rr_total / n,
        hit_rate=hits / n,
        coverage=len(recommended_items) / split.n_items,
        n_predictions=n,
        recall_cold=(recall_cold / n_cold) if n_cold else 0.0,
        mrr_cold=(rr_cold / n_cold) if n_cold else 0.0,
        n_cold=n_cold,
    )
