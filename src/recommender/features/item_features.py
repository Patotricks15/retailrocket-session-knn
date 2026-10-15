"""Item-level features: popularity and catalogue category.

``item_popularity_count`` is an interaction aggregate and therefore accepts an
explicit ``cutoff_order`` so the point-in-time rule can be asserted directly
in tests using a strict ``<`` comparison, exactly like the user/interaction
features. ``item_category`` is a catalogue attribute: it exists before any
interaction and never needs a cut-off.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _apply_cutoff(frame: pd.DataFrame, cutoff_order: int | None) -> pd.DataFrame:
    """Keep only events strictly before the cut-off position."""
    if cutoff_order is None:
        return frame
    return frame.loc[frame["event_order"] < cutoff_order]


def item_popularity_count(
    interactions: pd.DataFrame,
    n_items: int,
    cutoff_order: int | None = None,
) -> pd.Series:
    """Count training-session events per item, observable before the cut-off.

    Args:
        interactions: Training events carrying ``item_index`` and
            ``event_order``.
        n_items: Size of the item catalogue, so unseen items get an explicit
            zero rather than being absent from the index.
        cutoff_order: Strict point-in-time boundary; ``None`` uses every row
            passed in (the final, "at serving time" snapshot).

    Returns:
        A dense ``float32`` series indexed by ``item_index`` in
        ``[0, n_items)``.
    """
    frame = _apply_cutoff(interactions, cutoff_order)
    counts = frame.groupby("item_index").size()

    full = np.zeros(n_items, dtype=np.float32)
    full[counts.index.to_numpy()] = counts.to_numpy()

    series = pd.Series(full, index=pd.RangeIndex(n_items, name="item_index"))
    series.name = "item_popularity_count"
    return series


def item_category(
    catalogue: pd.DataFrame,
    item_ids: np.ndarray,
) -> pd.Series:
    """Category id per item, aligned to internal item indices.

    Items absent from the properties changelog (never observed in the
    catalogue snapshot) fall back to ``"unknown"`` further downstream via the
    declared fill policy, not here, so the raw series can still carry NaN.
    """
    lookup = catalogue.set_index("itemid")["categoryid"]
    values = lookup.reindex(item_ids)

    series = pd.Series(values.to_numpy(), index=pd.RangeIndex(len(item_ids), name="item_index"))
    series.name = "item_category"
    return series
