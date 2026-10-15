#!/usr/bin/env python3
"""Fetch the Retailrocket e-commerce dataset or generate a synthetic fixture.

The synthetic generator is the default so the repository is runnable from a
clean checkout with no Kaggle account. It reproduces the real schema --
``events.csv`` (view/addtocart/transaction), ``category_tree.csv`` and an
item-properties changelog -- and injects the structure a session-kNN model
depends on: bursty sessions separated by long idle gaps, a long-tailed item
popularity distribution, and visitors who revisit a handful of taste clusters
across multiple sessions.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
KAGGLE_DATASET = "retailrocket/ecommerce-dataset"

EVENT_TYPES = ["view", "addtocart", "transaction"]


def _make_category_tree(n_categories: int, rng: np.random.Generator) -> pd.DataFrame:
    """A shallow two-level category tree: a handful of roots, many leaves."""
    n_roots = max(3, n_categories // 8)
    categoryid = np.arange(1, n_categories + 1)
    parentid = np.full(n_categories, -1, dtype=object)
    parentid[n_roots:] = rng.integers(1, n_roots + 1, size=n_categories - n_roots)
    parentid[:n_roots] = None
    return pd.DataFrame({"categoryid": categoryid, "parentid": parentid})


def generate_synthetic(
    n_visitors: int = 1200,
    n_items: int = 400,
    n_categories: int = 40,
    seed: int = 20260115,
    span_days: int = 60,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build a Retailrocket-shaped fixture with realistic session structure.

    Each visitor is assigned a latent taste cluster and a number of sessions
    spread across ``span_days``; within a session, events are bursty (small
    inter-event gaps of seconds to a few minutes) while the gap *between*
    sessions is always well above the 30-minute sessionization threshold, so
    the synthetic data exercises the segmentation rule meaningfully instead of
    producing one session per visitor by construction.

    Args:
        n_visitors: Number of distinct ``visitorid`` values.
        n_items: Number of distinct ``itemid`` values.
        n_categories: Number of leaf+root categories.
        seed: Seed controlling the whole generation.
        span_days: Time range the event log is spread over.

    Returns:
        ``(events, category_tree, item_properties)`` frames matching the
        Kaggle schema.
    """
    rng = np.random.default_rng(seed)

    n_clusters = 6
    item_cluster = rng.integers(0, n_clusters, size=n_items)
    visitor_cluster = rng.integers(0, n_clusters, size=n_visitors)

    # Long-tailed popularity: a few items dominate, as in the real catalogue.
    popularity = rng.pareto(1.3, size=n_items) + 1.0
    popularity = popularity / popularity.sum()

    category_tree = _make_category_tree(n_categories, rng)
    item_category = rng.integers(1, n_categories + 1, size=n_items)

    start_ms = 1_430_000_000_000  # 2015-04-26, matching the real dataset's era
    span_ms = span_days * 24 * 3_600 * 1000

    event_records: list[dict[str, object]] = []
    property_records: list[dict[str, object]] = []
    transaction_seq = 0

    for visitor in range(1, n_visitors + 1):
        cluster = visitor_cluster[visitor - 1]
        affinity = np.where(item_cluster == cluster, 3.2, 0.3)
        weights = affinity * popularity
        weights = weights / weights.sum()

        n_sessions = int(rng.integers(1, 6))
        session_starts = np.sort(rng.integers(0, span_ms, size=n_sessions))

        for session_start in session_starts:
            session_len = int(rng.integers(1, 12))
            session_len = min(session_len, n_items)
            chosen = rng.choice(n_items, size=session_len, replace=False, p=weights)

            # Bursty intra-session gaps: seconds to a few minutes (<< 30 min).
            gaps = rng.integers(5_000, 400_000, size=session_len)
            gaps[0] = 0
            offsets = np.cumsum(gaps)

            in_cart = False
            for position, item_idx in enumerate(chosen):
                ts = start_ms + int(session_start) + int(offsets[position])
                in_cluster = item_cluster[item_idx] == cluster
                p_cart = 0.28 if in_cluster else 0.08
                p_buy = 0.5 if in_cluster else 0.15

                event = "view"
                transactionid = ""
                if position > 0 and not in_cart and rng.random() < p_cart:
                    event = "addtocart"
                    in_cart = True
                elif in_cart and rng.random() < p_buy:
                    event = "transaction"
                    transaction_seq += 1
                    transactionid = str(transaction_seq)
                    in_cart = False

                event_records.append(
                    {
                        "timestamp": ts,
                        "visitorid": visitor,
                        "event": event,
                        "itemid": int(item_idx) + 1,
                        "transactionid": transactionid,
                    }
                )

    for item_idx in range(n_items):
        property_records.append(
            {
                "timestamp": start_ms,
                "itemid": item_idx + 1,
                "property": "categoryid",
                "value": int(item_category[item_idx]),
            }
        )
        property_records.append(
            {
                "timestamp": start_ms,
                "itemid": item_idx + 1,
                "property": "available",
                "value": "1",
            }
        )

    events = pd.DataFrame.from_records(event_records).sort_values(
        "timestamp", kind="mergesort"
    ).reset_index(drop=True)
    item_properties = pd.DataFrame.from_records(property_records)
    return events, category_tree, item_properties


def download_from_kaggle(target: Path) -> None:
    """Download the real dataset using credentials from the environment."""
    if not os.environ.get("KAGGLE_USERNAME") or not os.environ.get("KAGGLE_KEY"):
        raise SystemExit(
            "KAGGLE_USERNAME and KAGGLE_KEY must be set in the environment.\n"
            "Credentials are never read from or written to this repository."
        )

    target.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "-m", "kaggle", "datasets", "download",
        "-d", KAGGLE_DATASET, "-p", str(target), "--unzip",
    ]
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as exc:
        raise SystemExit("The 'kaggle' package is not installed: pip install kaggle") from exc
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"Kaggle download failed with exit code {exc.returncode}") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--synthetic", action="store_true",
        help="Generate the offline fixture (default).",
    )
    source.add_argument(
        "--kaggle", action="store_true",
        help="Download the real dataset using KAGGLE_USERNAME/KAGGLE_KEY.",
    )
    parser.add_argument("--visitors", type=int, default=1200)
    parser.add_argument("--items", type=int, default=400)
    parser.add_argument("--categories", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260115)
    parser.add_argument("--span-days", type=int, default=60)
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if args.kaggle:
        download_from_kaggle(DATA_DIR)
        print(f"Downloaded {KAGGLE_DATASET} into {DATA_DIR}")
        return 0

    events, category_tree, item_properties = generate_synthetic(
        args.visitors, args.items, args.categories, args.seed, args.span_days
    )
    events.to_csv(DATA_DIR / "events.csv", index=False)
    category_tree.to_csv(DATA_DIR / "category_tree.csv", index=False)
    item_properties.to_csv(DATA_DIR / "item_properties.csv", index=False)

    print(
        f"Synthetic fixture written to {DATA_DIR}\n"
        f"  visitors={args.visitors} items={args.items} events={len(events)}\n"
        f"  transactions={(events['event'] == 'transaction').sum()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
