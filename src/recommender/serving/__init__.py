"""Serving: on-the-fly session-kNN scoring plus a precomputed popularity fallback."""

from __future__ import annotations

from recommender.serving.handler import lambda_handler, parse_request
from recommender.serving.topn import build_popularity_fallback_payload

__all__ = ["build_popularity_fallback_payload", "lambda_handler", "parse_request"]
