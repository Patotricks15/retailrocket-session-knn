"""Session-based k-Nearest-Neighbours: the primary, fully classical method.

The current (partial) session is represented as a set of recently viewed
items. At prediction time the ``k`` most similar *past* sessions are
retrieved from an inverted item->session index, each is weighted by how
similar it is to the query session and by how recent it is, and candidate
items are scored by a recency-weighted vote from those neighbours:

    sim(S, N) = |S ∩ N| / sqrt(|S| |N|)          (cosine on binary vectors)
    sim(S, N) = |S ∩ N| / |S ∪ N|                (Jaccard -- ablation)
    w(N)      = exp(-lambda * dt)                (recency decay, dt in hours)
    s(i, S)   = sum_{N in neighbours(S)} sim(S, N) * w(N) * 1[i in N]

No gradient descent, no embeddings: everything here is counting, set
arithmetic and a partial sort.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

MS_PER_HOUR = 1000 * 60 * 60


def _cosine_similarity(query: set[int], neighbour: set[int]) -> float:
    if not query or not neighbour:
        return 0.0
    intersection = len(query & neighbour)
    if intersection == 0:
        return 0.0
    return intersection / np.sqrt(len(query) * len(neighbour))


def _jaccard_similarity(query: set[int], neighbour: set[int]) -> float:
    if not query or not neighbour:
        return 0.0
    intersection = len(query & neighbour)
    if intersection == 0:
        return 0.0
    union = len(query) + len(neighbour) - intersection
    return intersection / union


SIMILARITY_FUNCTIONS = {
    "cosine": _cosine_similarity,
    "jaccard": _jaccard_similarity,
}


@dataclass
class SessionKNNRecommender:
    """Nearest-neighbour recommender over historical sessions.

    Args:
        k: Number of nearest neighbour sessions retrieved per query.
        sample_size: Upper bound on candidate neighbour sessions examined per
            query (the most recent ones sharing at least one item with the
            query), keeping scoring cheap as the training log grows without
            touching the result for any query whose true candidate pool is
            smaller than this bound.
        decay_lambda: Recency decay rate applied to neighbour sessions,
            ``w(N) = exp(-decay_lambda * dt)`` with ``dt`` in hours. Zero
            disables decay (a pure similarity vote).
        similarity: ``"cosine"`` (primary) or ``"jaccard"`` (ablation).
    """

    k: int = 100
    sample_size: int = 500
    decay_lambda: float = 0.01
    similarity: str = "cosine"

    n_items: int = 0
    session_items: dict[str, set[int]] = field(default_factory=dict)
    session_start_ts: dict[str, int] = field(default_factory=dict)
    item_to_sessions: dict[int, list[str]] = field(default_factory=dict)
    _sim_fn: object = None

    def __post_init__(self) -> None:
        if self.similarity not in SIMILARITY_FUNCTIONS:
            raise ValueError(
                f"Unknown similarity '{self.similarity}'; expected one of "
                f"{sorted(SIMILARITY_FUNCTIONS)}"
            )
        self._sim_fn = SIMILARITY_FUNCTIONS[self.similarity]

    def fit(self, split) -> SessionKNNRecommender:
        """Index every training session as a bag of items.

        Args:
            split: A :class:`recommender.data.splitting.SessionSplit`; only
                ``split.train`` is indexed, so neighbours are always drawn
                from sessions that chronologically precede validation/test.

        Returns:
            ``self``, fitted.

        Raises:
            ValueError: If the training partition is empty.
        """
        train = split.train
        if train.empty:
            raise ValueError("Cannot fit SessionKNN on an empty training partition.")

        self.n_items = split.n_items
        self.session_items = {
            sid: set(map(int, items))
            for sid, items in train.groupby("session_id")["item_index"]
        }
        self.session_start_ts = (
            train.groupby("session_id")["session_start_ts"].first().astype(np.int64).to_dict()
        )

        item_to_sessions: dict[int, list[str]] = {}
        # Sort sessions by start time so "most recent" truncation downstream
        # (``sample_size``) keeps the freshest candidates deterministically.
        ordered_sessions = sorted(self.session_items, key=lambda sid: self.session_start_ts[sid])
        for sid in ordered_sessions:
            for item in self.session_items[sid]:
                item_to_sessions.setdefault(item, []).append(sid)
        self.item_to_sessions = item_to_sessions
        return self

    def _candidate_sessions(self, query_items: set[int]) -> list[str]:
        """Union of sessions sharing >=1 item with the query, most recent first."""
        candidates: set[str] = set()
        for item in query_items:
            candidates.update(self.item_to_sessions.get(item, ()))
        if not candidates:
            return []
        ranked = sorted(candidates, key=lambda sid: self.session_start_ts[sid], reverse=True)
        return ranked[: self.sample_size]

    def neighbours(
        self, query_items: set[int], query_start_ts: int
    ) -> list[tuple[str, float, float]]:
        """Return the ``k`` nearest neighbour sessions as ``(id, sim, weight)``."""
        if not self._sim_fn:  # pragma: no cover - defensive, set in __post_init__
            raise RuntimeError("Model is not initialised.")

        scored = []
        for sid in self._candidate_sessions(query_items):
            sim = self._sim_fn(query_items, self.session_items[sid])
            if sim <= 0.0:
                continue
            age_hours = max(0.0, (query_start_ts - self.session_start_ts[sid]) / MS_PER_HOUR)
            weight = float(np.exp(-self.decay_lambda * age_hours))
            scored.append((sid, sim, weight))

        scored.sort(key=lambda row: row[1], reverse=True)
        return scored[: self.k]

    def score(self, query_items: set[int], query_start_ts: int) -> np.ndarray:
        """Return the recency-weighted neighbour vote over the full catalogue."""
        scores = np.zeros(self.n_items, dtype=np.float64)
        for sid, sim, weight in self.neighbours(query_items, query_start_ts):
            vote = sim * weight
            for item in self.session_items[sid]:
                scores[item] += vote
        return scores

    def recommend(
        self,
        query_items: set[int],
        query_start_ts: int,
        k: int = 10,
        exclude: set[int] | None = None,
    ) -> np.ndarray:
        """Return the top-``k`` candidate item indices for a partial session.

        Args:
            query_items: Items observed so far in the current session.
            query_start_ts: Start timestamp of the current session.
            k: List length.
            exclude: Items to filter out; normally ``query_items`` itself, so
                the model never "recommends" what the visitor just viewed.

        Returns:
            Item indices ordered by descending score.
        """
        scores = self.score(query_items, query_start_ts)
        if exclude:
            scores = scores.copy()
            scores[list(exclude)] = -np.inf

        k = min(k, scores.shape[0])
        top = np.argpartition(-scores, k - 1)[:k]
        ordered = top[np.argsort(-scores[top])]
        return ordered[scores[ordered] > -np.inf]

    def _check_fitted(self) -> None:
        if not self.session_items:
            raise RuntimeError("Model is not fitted; call fit() first.")
