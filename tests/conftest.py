"""Shared pytest fixtures built from the synthetic dataset."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"

from recommender.data import (  # noqa: E402
    build_item_catalogue,
    index_items,
    load_category_tree,
    load_events,
    load_item_properties,
    sessionize,
    temporal_session_split,
)
from recommender.features import build_features  # noqa: E402


def _ensure_dataset() -> None:
    """Generate the synthetic fixture on demand so tests are self-contained."""
    if (DATA_DIR / "events.csv").exists() and (DATA_DIR / "item_properties.csv").exists():
        return

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from download_data import generate_synthetic

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    events, category_tree, item_properties = generate_synthetic(
        n_visitors=500, n_items=180, n_categories=20, seed=7
    )
    events.to_csv(DATA_DIR / "events.csv", index=False)
    category_tree.to_csv(DATA_DIR / "category_tree.csv", index=False)
    item_properties.to_csv(DATA_DIR / "item_properties.csv", index=False)


@pytest.fixture(scope="session")
def raw_events() -> pd.DataFrame:
    _ensure_dataset()
    return load_events(DATA_DIR / "events.csv")


@pytest.fixture(scope="session")
def catalogue() -> pd.DataFrame:
    _ensure_dataset()
    props = load_item_properties(DATA_DIR / "item_properties.csv")
    return build_item_catalogue(props)


@pytest.fixture(scope="session")
def category_tree() -> pd.DataFrame:
    _ensure_dataset()
    return load_category_tree(DATA_DIR / "category_tree.csv")


@pytest.fixture(scope="session")
def event_data(raw_events: pd.DataFrame):
    return index_items(raw_events)


@pytest.fixture(scope="session")
def sessionized(event_data):
    return sessionize(event_data.frame)


@pytest.fixture(scope="session")
def split(sessionized, event_data):
    return temporal_session_split(sessionized, n_items=event_data.n_items)


@pytest.fixture(scope="session")
def feature_store(split, catalogue, event_data):
    return build_features(split, catalogue, event_data.item_ids)
