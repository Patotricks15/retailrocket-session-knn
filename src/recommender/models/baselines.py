"""Baselines the primary method must beat to justify its complexity.

Both baselines share the session-kNN call signature
(``recommend(query_items, query_start_ts, k, exclude)``) even though neither
uses ``query_start_ts``, so the evaluation harness can treat every model
polymorphically.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PopularityRecommender:
    """Recommends the globally most-viewed training items to every session.

    This is the baseline naive "just show the bestsellers" systems default
    to, and the one session-kNN has to beat to justify any complexity at all.
    """

    item_scores: np.ndarray | None = None
    n_items: int = 0

    def fit(self, split) -> PopularityRecommender:
        self.n_items = split.n_items
        counts = np.zeros(self.n_items, dtype=np.float64)
        items = split.train["item_index"].to_numpy()
        np.add.at(counts, items, 1.0)
        self.item_scores = counts
        return self

    def recommend(
        self,
        query_items: set[int],
        query_start_ts: int,  # noqa: ARG002
        k: int = 10,
        exclude: set[int] | None = None,
    ) -> np.ndarray:
        if self.item_scores is None:
            raise RuntimeError("Model is not fitted; call fit() first.")

        scores = self.item_scores.copy()
        if exclude:
            scores[list(exclude)] = -np.inf

        k = min(k, scores.shape[0])
        top = np.argpartition(-scores, k - 1)[:k]
        ordered = top[np.argsort(-scores[top])]
        return ordered[scores[ordered] > -np.inf]


@dataclass
class RandomRecommender:
    """Uniform random ordering; the floor every other model must clear."""

    random_state: int = 0
    n_items: int = 0

    def fit(self, split) -> RandomRecommender:
        self.n_items = split.n_items
        return self

    def recommend(
        self,
        query_items: set[int],
        query_start_ts: int,
        k: int = 10,
        exclude: set[int] | None = None,
    ) -> np.ndarray:
        seed = self.random_state + int(query_start_ts % 1_000_003) + sum(query_items)
        rng = np.random.default_rng(seed)
        scores = rng.random(self.n_items)
        if exclude:
            scores[list(exclude)] = -np.inf

        k = min(k, scores.shape[0])
        top = np.argpartition(-scores, k - 1)[:k]
        ordered = top[np.argsort(-scores[top])]
        return ordered[scores[ordered] > -np.inf]
