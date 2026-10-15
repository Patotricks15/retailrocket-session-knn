"""Command line interface: train, evaluate and recommend.

Hyperparameter defaults are the values selected by a grid search on the
*validation* partition (see ``scripts/tune.py``). The test partition is
touched only by ``evaluate --partition test``, and only once, which is what
keeps the reported numbers honest.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import pandas as pd

from recommender.data import (
    build_item_catalogue,
    index_items,
    load_events,
    load_item_properties,
    sessionize,
    temporal_session_split,
)
from recommender.evaluation import evaluate
from recommender.models import PopularityRecommender, RandomRecommender, SessionKNNRecommender

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = PROJECT_ROOT / "artifacts"
DEFAULT_EVENTS = PROJECT_ROOT / "data" / "events.csv"
DEFAULT_PROPERTIES = PROJECT_ROOT / "data" / "item_properties.csv"

# Selected on the validation partition; see `scripts/tune.py`.
BEST_KNN = {"k": 100, "sample_size": 500, "decay_lambda": 0.0, "similarity": "cosine"}


def _load_split(events_path: Path):
    events = load_events(events_path)
    data = index_items(events)
    sessionized = sessionize(data.frame)
    split = temporal_session_split(sessionized, n_items=data.n_items)
    return data, split


def _build_models(seed: int) -> dict:
    return {
        "Random": RandomRecommender(random_state=seed),
        "Popularity": PopularityRecommender(),
        "SessionKNN": SessionKNNRecommender(**BEST_KNN),
    }


def cmd_train(args: argparse.Namespace) -> int:
    data, split = _load_split(args.events)
    counts = split.session_counts()
    print(f"items={split.n_items}  sessions(train/val/test)={counts}")

    model = SessionKNNRecommender(**BEST_KNN).fit(split)

    ARTIFACTS.mkdir(exist_ok=True)
    with (ARTIFACTS / "session_knn.pkl").open("wb") as handle:
        pickle.dump({"model": model, "item_ids": data.item_ids}, handle)

    print(f"SessionKNN fitted: {len(model.session_items)} indexed training sessions")
    print(f"saved -> {ARTIFACTS / 'session_knn.pkl'}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    _, split = _load_split(args.events)

    rows = []
    for name, model in _build_models(args.seed).items():
        model.fit(split)
        report = evaluate(model, split, k=args.k, partition=args.partition, model_name=name)
        rows.append(report.as_row())
        print(
            f"{name:12} recall@{args.k}={report.recall:.4f}  mrr={report.mrr:.4f}  "
            f"coverage={report.coverage:.3f}  cold_recall={report.recall_cold:.4f}"
        )

    ARTIFACTS.mkdir(exist_ok=True)
    frame = pd.DataFrame(rows).sort_values("recall", ascending=False)
    frame.to_csv(ARTIFACTS / "results.csv", index=False)
    print(f"\nsaved -> {ARTIFACTS / 'results.csv'}")
    return 0


def cmd_recommend(args: argparse.Namespace) -> int:
    data, split = _load_split(args.events)

    frame = split.test if args.partition == "test" else split.validation
    session_rows = frame.loc[frame["session_id"] == args.session_id]
    if session_rows.empty:
        raise SystemExit(
            f"session_id {args.session_id!r} not found in the {args.partition} partition."
        )

    session_rows = session_rows.sort_values("session_position")
    items = session_rows["item_index"].to_numpy()
    prefix_len = min(args.prefix_len, len(items) - 1) if len(items) > 1 else len(items)
    prefix = set(int(i) for i in items[:prefix_len]) if prefix_len else set(int(i) for i in items)
    start_ts = int(session_rows["session_start_ts"].iloc[0])

    model = SessionKNNRecommender(**BEST_KNN).fit(split)
    ranked = model.recommend(prefix, start_ts, k=args.k, exclude=prefix)

    titles = {}
    if args.properties.exists():
        catalogue = build_item_catalogue(load_item_properties(args.properties))
        titles = dict(zip(catalogue["itemid"], catalogue["categoryid"], strict=False))

    payload = [
        {
            "rank": rank,
            "itemid": int(data.item_ids[int(item_index)]),
            "categoryid": titles.get(int(data.item_ids[int(item_index)]), "?"),
        }
        for rank, item_index in enumerate(ranked, start=1)
    ]
    print(json.dumps(payload, indent=2, default=str))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="retailrocket-knn", description=__doc__)
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--seed", type=int, default=42)
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Fit SessionKNN and persist the artifact.")
    train.set_defaults(func=cmd_train)

    ev = sub.add_parser("evaluate", help="Benchmark SessionKNN against the baselines.")
    ev.add_argument("--k", type=int, default=20)
    ev.add_argument("--partition", choices=["validation", "test"], default="test")
    ev.set_defaults(func=cmd_evaluate)

    rec = sub.add_parser("recommend", help="Top-k list continuing one partial session.")
    rec.add_argument("--session-id", type=str, required=True)
    rec.add_argument("--partition", choices=["validation", "test"], default="test")
    rec.add_argument("--prefix-len", type=int, default=1)
    rec.add_argument("--k", type=int, default=10)
    rec.add_argument("--properties", type=Path, default=DEFAULT_PROPERTIES)
    rec.set_defaults(func=cmd_recommend)

    args = parser.parse_args(argv)
    if not args.events.exists():
        raise SystemExit(
            f"{args.events} not found. Run: python scripts/download_data.py --synthetic"
        )
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
