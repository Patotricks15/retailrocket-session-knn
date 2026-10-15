"""Loading of the Retailrocket raw/synthetic event stream and catalogue.

The raw Kaggle dataset ships three files:

``events.csv``
    ``timestamp,visitorid,event,itemid,transactionid`` where ``event`` is one
    of ``view``, ``addtocart`` or ``transaction``. ``visitorid`` is a cookie,
    not a durable account: the same person can appear under several ids and
    the same id can be shared by several people, which is precisely why this
    project treats the *session* rather than the *visitor* as the unit of
    personalization.

``item_properties_part1.csv`` / ``item_properties_part2.csv``
    ``timestamp,itemid,property,value``, a changelog of item attributes
    (including ``categoryid`` and ``available``) sampled over time.

``category_tree.csv``
    ``categoryid,parentid``.

``scripts/download_data.py`` produces a synthetic fixture with the same
schema by default, so the repository is runnable with no Kaggle account.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

VALID_EVENTS = {"view", "addtocart", "transaction"}
REQUIRED_EVENT_COLUMNS = {"timestamp", "visitorid", "event", "itemid"}


@dataclass(frozen=True)
class EventData:
    """Raw events with contiguous internal item indices.

    Attributes:
        frame: Events sorted by timestamp, carrying ``item_index``.
        item_ids: Original ``itemid`` values, positionally aligned to
            ``item_index``.
    """

    frame: pd.DataFrame
    item_ids: np.ndarray

    @property
    def n_items(self) -> int:
        return int(self.item_ids.shape[0])

    def __len__(self) -> int:
        return int(self.frame.shape[0])


def load_events(path: str | Path) -> pd.DataFrame:
    """Read the raw (or synthetic) events file and validate its contract.

    Args:
        path: Path to ``events.csv``.

    Returns:
        The raw frame, sorted by ``timestamp``.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If required columns are missing or an event type is
            outside ``VALID_EVENTS``.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run 'python scripts/download_data.py --synthetic' first."
        )

    frame = pd.read_csv(path)
    missing = REQUIRED_EVENT_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    bad_events = set(frame["event"].unique()) - VALID_EVENTS
    if bad_events:
        raise ValueError(f"{path} contains unknown event types: {sorted(bad_events)}")

    return frame.sort_values("timestamp", kind="mergesort").reset_index(drop=True)


def load_category_tree(path: str | Path) -> pd.DataFrame:
    """Read ``category_tree.csv``: ``categoryid,parentid``."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run 'python scripts/download_data.py --synthetic' first."
        )
    frame = pd.read_csv(path)
    required = {"categoryid", "parentid"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    return frame


def load_item_properties(path: str | Path) -> pd.DataFrame:
    """Read an item-properties changelog: ``timestamp,itemid,property,value``."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run 'python scripts/download_data.py --synthetic' first."
        )
    frame = pd.read_csv(path)
    required = {"timestamp", "itemid", "property", "value"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    return frame


def build_item_catalogue(item_properties: pd.DataFrame) -> pd.DataFrame:
    """Collapse the properties changelog into one row per item.

    Only the catalogue-level attributes (``categoryid``) are kept; this is
    metadata known independently of any interaction, so it is available even
    for cold-start items. The *latest* reported value per item is used, which
    mirrors how a real catalogue snapshot would be joined at training time.

    Args:
        item_properties: Raw changelog from :func:`load_item_properties`.

    Returns:
        A frame with ``itemid`` and ``categoryid`` columns.
    """
    categories = item_properties.loc[item_properties["property"] == "categoryid"].copy()
    if categories.empty:
        return pd.DataFrame({"itemid": pd.Series(dtype=np.int64),
                              "categoryid": pd.Series(dtype=np.int64)})

    categories = categories.sort_values(["itemid", "timestamp"], kind="mergesort")
    latest = categories.groupby("itemid", as_index=False).tail(1)
    latest = latest[["itemid", "value"]].rename(columns={"value": "categoryid"})
    latest["categoryid"] = pd.to_numeric(latest["categoryid"], errors="coerce").astype("Int64")
    return latest.reset_index(drop=True)


def index_items(events: pd.DataFrame) -> EventData:
    """Assign contiguous ``item_index`` values and a global chronological order.

    Args:
        events: Output of :func:`load_events`.

    Returns:
        :class:`EventData` with ``item_index`` and ``event_order`` columns,
        where ``event_order`` is a strict, gap-free chronological position
        used as the point-in-time cut-off everywhere downstream.
    """
    frame = events.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    item_ids = np.sort(frame["itemid"].unique())
    item_map = {iid: idx for idx, iid in enumerate(item_ids)}
    frame["item_index"] = frame["itemid"].map(item_map).astype(np.int32)
    frame["event_order"] = np.arange(frame.shape[0], dtype=np.int64)
    return EventData(frame=frame, item_ids=item_ids)
