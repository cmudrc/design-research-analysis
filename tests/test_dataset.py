from __future__ import annotations

import pytest

from design_research_analysis.dataset import (
    generate_codebook,
    profile_dataframe,
    validate_dataframe,
)

pd = pytest.importorskip("pandas")


def test_profile_dataframe_reports_numeric_summary() -> None:
    df = pd.DataFrame(
        {
            "score": [1, 2, 3, 4, 5, 6, 7, 100],
            "group": ["a", "a", "b", "b", "c", "c", "d", "d"],
        }
    )

    profile = profile_dataframe(df)

    assert profile["n_rows"] == 8
    assert profile["columns"]["score"]["outlier_count_iqr"] == 1
    assert "sample_values" in profile["columns"]["group"]


def test_profile_dataframe_warns_on_high_cardinality() -> None:
    df = pd.DataFrame({"label": [f"id_{idx}" for idx in range(6)]})

    profile = profile_dataframe(df, max_categorical_levels=3)

    assert profile["warnings"]


def test_validate_dataframe_reports_errors_and_warnings() -> None:
    df = pd.DataFrame({"participant_id": [1, 1], "extra": [0, 1]})
    schema = {
        "participant_id": {"unique": True, "nullable": False},
        "age": {"required": True},
        "bad_rule": {"unknown": True},
    }

    result = validate_dataframe(df, schema)

    assert result["ok"] is False
    assert any("Required column 'age'" in error for error in result["errors"])
    assert any("unsupported schema key 'unknown'" in error for error in result["errors"])
    assert any("duplicate non-null values" in error for error in result["errors"])
    assert result["summary"]["unexpected_columns"] == ["extra"]
    assert result["warnings"]


def test_generate_codebook_preserves_order_and_descriptions() -> None:
    df = pd.DataFrame({"first": [1, 2], "second": ["x", "y"]})

    codebook = generate_codebook(df, descriptions={"second": "Condition label"})

    assert list(codebook["column"]) == ["first", "second"]
    assert list(codebook.columns) == [
        "column",
        "inferred_dtype",
        "nonnull_count",
        "missing_count",
        "missing_fraction",
        "n_unique",
        "example_values",
        "description",
    ]
    assert codebook.loc[1, "description"] == "Condition label"
    assert codebook.loc[0, "description"] == ""
