"""Feature pipeline: fits corpus-level features on train and augments any
partition with the row-local session/context features.

The same object is used by training and by the serving path (session-kNN
scores every candidate item; this is where the ``item_popularity_count`` and
``item_category`` side-information used for cold-start fallback and reranking
comes from), which is what makes offline/online parity testable rather than
aspirational.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from recommender.data.sessionize import session_summary
from recommender.features import (
    context_features,
    interaction_features,
    item_features,
    user_features,
)
from recommender.features.registry import validate_registry
from recommender.features.spec import FeatureSpec, load_feature_specs


@dataclass
class FeatureStore:
    """Materialized corpus-level features, fitted on the training partition only."""

    item_popularity_count: pd.Series
    item_category: pd.Series
    visitor_prior_session_count: pd.Series
    n_items: int
    specs: list[FeatureSpec] = field(default_factory=list)

    def item_vector(self, item_index: int) -> dict[str, float | str]:
        """Return the declared-fill item features for one internal item index."""
        popularity = float(self.item_popularity_count.get(item_index, 0.0))
        category = self.item_category.get(item_index, np.nan)
        return {
            "item_popularity_count": popularity,
            "item_category": "unknown" if pd.isna(category) else str(category),
        }

    def visitor_prior_sessions(self, visitorid: int) -> float:
        """Return the declared-fill visitor feature; 0.0 for an unseen cookie."""
        return float(self.visitor_prior_session_count.get(visitorid, 0.0))

    def augment_events(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Attach the row-local interaction/context features to any partition.

        These features need no fitted state: they are computed directly from
        each row's own session, so the exact same function is applied whether
        ``frame`` is the train, validation or test partition.
        """
        out = frame.copy()
        out["session_length_so_far"] = interaction_features.session_length_so_far(out)
        out["time_since_session_start"] = interaction_features.time_since_session_start(out)
        out["hour_of_day"] = context_features.hour_of_day(out)
        out["is_weekend"] = context_features.is_weekend(out)
        return out

    def to_frames(self) -> dict[str, pd.DataFrame]:
        """Export item-level features as a tidy frame for the paper tables."""
        items = pd.DataFrame(
            {
                "item_index": np.arange(self.n_items),
                "item_popularity_count": self.item_popularity_count.reindex(
                    range(self.n_items)
                ).fillna(0.0).to_numpy(),
                "item_category": self.item_category.reindex(range(self.n_items))
                .fillna("unknown")
                .astype(str)
                .to_numpy(),
            }
        )
        return {"items": items}


def build_features(
    split,
    catalogue: pd.DataFrame,
    item_ids: np.ndarray,
    spec_path: str | Path | None = None,
) -> FeatureStore:
    """Materialize every declared corpus-level feature from the train partition.

    Only ``split.train`` is passed to the interaction/visitor transforms,
    which is what enforces the ``fit_on: train_only`` contract. The catalogue
    feature reads item metadata directly because it is known before any
    interaction.

    Args:
        split: A :class:`recommender.data.splitting.SessionSplit`.
        catalogue: Item-category lookup (``itemid, categoryid``).
        item_ids: Original ``itemid`` values aligned to internal indices.
        spec_path: Optional override of ``conf/features.yaml``.

    Returns:
        A populated :class:`FeatureStore`.
    """
    specs = load_feature_specs(spec_path)
    validate_registry(specs)

    train = split.train
    train_sessions = session_summary(train)

    return FeatureStore(
        item_popularity_count=item_features.item_popularity_count(train, n_items=split.n_items),
        item_category=item_features.item_category(catalogue, item_ids),
        visitor_prior_session_count=user_features.visitor_prior_session_count(train_sessions),
        n_items=split.n_items,
        specs=specs,
    )
