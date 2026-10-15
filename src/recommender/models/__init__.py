"""Classical recommendation models: session-kNN and its baselines."""

from __future__ import annotations

from recommender.models.baselines import PopularityRecommender, RandomRecommender
from recommender.models.session_knn import SessionKNNRecommender

__all__ = ["PopularityRecommender", "RandomRecommender", "SessionKNNRecommender"]
