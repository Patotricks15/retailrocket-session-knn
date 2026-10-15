"""Registry binding declared feature names to their implementations."""

from __future__ import annotations

from collections.abc import Callable

from recommender.features import (
    context_features,
    interaction_features,
    item_features,
    user_features,
)
from recommender.features.spec import FeatureSpec

TRANSFORM_REGISTRY: dict[str, Callable[..., object]] = {
    "item_popularity_count": item_features.item_popularity_count,
    "item_category": item_features.item_category,
    "visitor_prior_session_count": user_features.visitor_prior_session_count,
    "session_length_so_far": interaction_features.session_length_so_far,
    "time_since_session_start": interaction_features.time_since_session_start,
    "hour_of_day": context_features.hour_of_day,
    "is_weekend": context_features.is_weekend,
}


def validate_registry(specs: list[FeatureSpec]) -> None:
    """Assert the declaration and the implementation agree in both directions.

    Args:
        specs: Parsed feature specifications.

    Raises:
        ValueError: If a declared feature has no implementation, or an
            implemented transform is never declared.
    """
    declared = {spec.transform for spec in specs}
    implemented = set(TRANSFORM_REGISTRY)

    undeclared = implemented - declared
    unimplemented = declared - implemented

    problems: list[str] = []
    if unimplemented:
        problems.append(f"declared but not implemented: {sorted(unimplemented)}")
    if undeclared:
        problems.append(f"implemented but not declared: {sorted(undeclared)}")

    if problems:
        raise ValueError("Feature registry mismatch -- " + "; ".join(problems))
