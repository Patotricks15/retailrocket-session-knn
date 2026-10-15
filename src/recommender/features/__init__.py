"""Declarative, point-in-time feature engineering for session-kNN."""

from __future__ import annotations

from recommender.features.pipeline import FeatureStore, build_features
from recommender.features.registry import TRANSFORM_REGISTRY, validate_registry
from recommender.features.spec import FeatureSpec, load_feature_specs

__all__ = [
    "TRANSFORM_REGISTRY",
    "FeatureSpec",
    "FeatureStore",
    "build_features",
    "load_feature_specs",
    "validate_registry",
]
