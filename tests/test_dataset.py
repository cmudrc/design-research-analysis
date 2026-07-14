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


def test_profile_dataframe_accepts_csv_path(tmp_path) -> None:
    csv_path = tmp_path / "dataset.csv"
    pd.DataFrame({"score": [1, 2], "group": ["a", "b"]}).to_csv(csv_path, index=False)

    profile = profile_dataframe(csv_path)

    assert profile["n_rows"] == 2
    assert set(profile["columns"]) == {"score", "group"}


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


def test_validate_dataframe_rule_variants() -> None:
    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "score": [10, 20, 30],
            "group": ["A", "B", "A"],
            "code": ["X-1", "X-2", "X-3"],
            "when": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
        }
    )
    schema = {
        "id": {"dtype": "integer", "nullable": False, "unique": True},
        "score": {"dtype": "numeric", "min": 0, "max": 100},
        "group": {"allowed": ["A", "B"]},
        "code": {"regex": r"X-\d"},
        "when": {"dtype": "datetime", "min": pd.Timestamp("2026-01-01")},
    }
    result = validate_dataframe(df, schema)
    assert result["ok"] is True


def test_validate_dataframe_accepts_json_path(tmp_path) -> None:
    json_path = tmp_path / "dataset.json"
    json_path.write_text('[{"participant_id": 1}, {"participant_id": 2}]', encoding="utf-8")

    result = validate_dataframe(json_path, {"participant_id": {"unique": True, "nullable": False}})

    assert result["ok"] is True


def test_validate_dataframe_dtype_and_bound_errors() -> None:
    df = pd.DataFrame(
        {
            "flag": ["yes", "no"],
            "score": ["high", "low"],
            "group": ["A", "C"],
            "code": ["bad", "X-2"],
        }
    )
    schema = {
        "flag": {"dtype": "boolean"},
        "score": {"min": 0, "max": 10},
        "group": {"allowed": ["A", "B"]},
        "code": {"regex": r"X-\d"},
        "unknown_dtype": {"dtype": "imaginary"},
    }
    result = validate_dataframe(df, schema)
    assert result["ok"] is False
    assert any("does not match declared dtype" in err for err in result["errors"])
    assert any("cannot use 'min'" in err for err in result["errors"])
    assert any("outside the allowed set" in err for err in result["errors"])
    assert any("fail regex validation" in err for err in result["errors"])


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


def test_generate_codebook_accepts_csv_path(tmp_path) -> None:
    csv_path = tmp_path / "codebook.csv"
    pd.DataFrame({"first": [1, 2], "second": ["x", "y"]}).to_csv(csv_path, index=False)

    codebook = generate_codebook(csv_path)

    assert list(codebook["column"]) == ["first", "second"]


def test_dataset_input_formats_and_errors(tmp_path) -> None:
    """Dataset coercion should support documented paths and reject malformed JSON."""
    tsv_path = tmp_path / "dataset.tsv"
    tsv_path.write_text("score\tgroup\n1\ta\n", encoding="utf-8")
    assert profile_dataframe(tsv_path)["n_rows"] == 1

    columnar_path = tmp_path / "columnar.json"
    columnar_path.write_text('{"score": [1, 2]}', encoding="utf-8")
    assert profile_dataframe(columnar_path)["n_rows"] == 2

    non_object_rows = tmp_path / "bad-rows.json"
    non_object_rows.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(ValueError, match="list of objects"):
        profile_dataframe(non_object_rows)

    scalar_path = tmp_path / "scalar.json"
    scalar_path.write_text("1", encoding="utf-8")
    with pytest.raises(ValueError, match="columnar object"):
        profile_dataframe(scalar_path)
    with pytest.raises(ValueError, match="Unsupported dataset input"):
        profile_dataframe(tmp_path / "dataset.txt")
    with pytest.raises(TypeError, match="pandas DataFrame"):
        profile_dataframe(object())  # type: ignore[arg-type]


def test_dataset_dtype_inference_covers_boolean_category_and_datetime() -> None:
    """Profiles should expose stable names for pandas extension dtypes."""
    frame = pd.DataFrame(
        {
            "flag": pd.Series([True, False], dtype="boolean"),
            "category": pd.Series(["a", "b"], dtype="category"),
            "when": pd.to_datetime(["2026-01-01", "2026-01-02"]),
            "value": pd.Series([1.5, 2.5], dtype="float64"),
        }
    )
    columns = profile_dataframe(frame)["columns"]
    assert columns["flag"]["inferred_dtype"] == "boolean"
    assert columns["category"]["inferred_dtype"] == "category"
    assert columns["when"]["inferred_dtype"] == "datetime"
    assert columns["value"]["inferred_dtype"] == "numeric"


def test_validate_dataframe_reports_every_rule_family() -> None:
    """Schema diagnostics should cover malformed declarations and value constraints."""
    frame = pd.DataFrame(
        {
            "score": [0, 20],
            "label": ["a", "bad"],
            "nullable": [1.0, None],
            "unknown_dtype": [1, 2],
        }
    )
    result = validate_dataframe(
        frame,
        {
            "not_a_mapping": "required",
            "score": {"min": 1, "max": 10, "regex": r"\d+"},
            "label": {"allowed": "a", "regex": r"[a-z]"},
            "nullable": {"nullable": False},
            "unknown_dtype": {"dtype": "imaginary"},
        },
    )
    errors = "\n".join(result["errors"])
    assert "must be a mapping" in errors
    assert "below the allowed minimum" in errors
    assert "above the allowed maximum" in errors
    assert "cannot use 'regex'" in errors
    assert "outside the allowed set" in errors
    assert "contains null values" in errors
    assert "unsupported dtype" in errors
