"""Schema contract tests for the declarative feature specification."""

from __future__ import annotations

import pytest

from recommender.features import load_feature_specs, validate_registry
from recommender.features.spec import VALID_FAMILIES, FeatureSpec


def test_registry_matches_declaration() -> None:
    """Every declared feature is implemented and vice versa."""
    validate_registry(load_feature_specs())


def test_every_feature_declares_a_known_family() -> None:
    for spec in load_feature_specs():
        assert spec.family in VALID_FAMILIES


def test_every_feature_documents_a_rationale() -> None:
    """A feature without a stated reason cannot be reviewed or ablated."""
    for spec in load_feature_specs():
        assert len(spec.rationale.strip()) > 20, f"{spec.name} has a thin rationale"


def test_item_and_user_aggregates_are_point_in_time_and_train_scoped() -> None:
    """Features that aggregate interaction history must declare both contracts."""
    for spec in load_feature_specs():
        if spec.fit_on == "train_only":
            assert spec.point_in_time, f"{spec.name} fits on train but is not point-in-time"
            assert spec.family in {"item", "user"}, (
                f"{spec.name} is train_only but not an item/user aggregate"
            )


def test_session_local_interaction_features_are_point_in_time() -> None:
    """Session-local features have no corpus fit, but still respect the cut-off."""
    for spec in load_feature_specs():
        if spec.family == "interaction":
            assert spec.point_in_time, f"{spec.name} is session-local but not point-in-time"
            assert spec.fit_on == "none"


def test_context_features_are_not_aggregates() -> None:
    """Context features are deterministic in the row's own timestamp."""
    for spec in load_feature_specs():
        if spec.family == "context":
            assert not spec.point_in_time
            assert spec.fit_on == "none"


def test_catalogue_features_are_not_train_scoped() -> None:
    """Item metadata exists before any interaction, so it is cold-start safe."""
    for spec in load_feature_specs():
        if spec.fit_on == "catalogue":
            assert not spec.point_in_time


def test_invalid_family_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown family"):
        FeatureSpec(
            name="bad", family="nonsense", dtype="float32", transform="t",
            params={}, fit_on="train_only", point_in_time=True, fill={},
            rationale="long enough rationale for the validator to accept it",
        )


def test_missing_rationale_is_rejected() -> None:
    with pytest.raises(ValueError, match="rationale"):
        FeatureSpec(
            name="bad", family="item", dtype="float32", transform="t",
            params={}, fit_on="catalogue", point_in_time=False, fill={},
            rationale="   ",
        )


def test_feature_store_has_no_nulls_and_correct_dtypes(feature_store) -> None:
    """Declared fill policies must leave no missing values downstream."""
    frames = feature_store.to_frames()

    assert frames["items"].isna().sum().sum() == 0
    assert len(frames["items"]) == feature_store.n_items


def test_unknown_entities_fall_back_to_declared_defaults(feature_store) -> None:
    """Cold entities must not raise; they must return the declared fill value."""
    ghost_item = feature_store.n_items + 999
    ghost_visitor = -1

    vector = feature_store.item_vector(ghost_item)
    assert vector["item_popularity_count"] == 0.0
    assert vector["item_category"] == "unknown"
    assert feature_store.visitor_prior_sessions(ghost_visitor) == 0.0


def test_augmented_events_have_every_declared_row_local_feature(feature_store, split) -> None:
    augmented = feature_store.augment_events(split.validation)
    for column in (
        "session_length_so_far",
        "time_since_session_start",
        "hour_of_day",
        "is_weekend",
    ):
        assert column in augmented.columns
        assert augmented[column].isna().sum() == 0
