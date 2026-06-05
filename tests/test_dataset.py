from __future__ import annotations

import json

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


def test_profile_dataframe_file_formats_and_dtype_edges(tmp_path) -> None:
    tsv_path = tmp_path / "dataset.tsv"
    pd.DataFrame(
        {
            "flag": [True, False],
            "when": pd.to_datetime(["2026-01-01", "2026-01-02"]),
            "category": pd.Series(["a", "b"], dtype="category"),
            "ratio": [0.1, 0.2],
        }
    ).to_csv(tsv_path, sep="\t", index=False)

    profile = profile_dataframe(tsv_path)
    assert profile["columns"]["flag"]["inferred_dtype"] == "boolean"
    assert profile["columns"]["when"]["inferred_dtype"] == "string"
    assert profile["columns"]["ratio"]["inferred_dtype"] == "numeric"

    typed_profile = profile_dataframe(
        pd.DataFrame(
            {
                "category": pd.Series(["a", "b"], dtype="category"),
                "when": pd.to_datetime(["2026-01-01", "2026-01-02"]),
            }
        )
    )
    assert typed_profile["columns"]["category"]["inferred_dtype"] == "category"
    assert typed_profile["columns"]["when"]["inferred_dtype"] == "datetime"

    columnar_json = tmp_path / "columnar.json"
    columnar_json.write_text(json.dumps({"score": [1, 2], "group": ["a", "b"]}), encoding="utf-8")
    assert profile_dataframe(columnar_json)["n_rows"] == 2

    bad_rows = tmp_path / "bad-list.json"
    bad_rows.write_text(json.dumps([1, 2]), encoding="utf-8")
    with pytest.raises(ValueError, match="list of objects"):
        profile_dataframe(bad_rows)

    bad_payload = tmp_path / "bad-payload.json"
    bad_payload.write_text(json.dumps("not rows"), encoding="utf-8")
    with pytest.raises(ValueError, match="JSON input"):
        profile_dataframe(bad_payload)

    unsupported = tmp_path / "dataset.xlsx"
    unsupported.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported dataset input format"):
        profile_dataframe(unsupported)

    with pytest.raises(TypeError, match="Dataset input"):
        profile_dataframe([{"score": 1}])  # type: ignore[arg-type]


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
            "bad_rule": [1, 2],
            "nullable": [1, None],
            "flag": ["yes", "no"],
            "score": ["high", "low"],
            "low_score": [-1, 2],
            "high_score": [1, 20],
            "group": ["A", "C"],
            "literal": ["A", "B"],
            "scalar_allowed": [1, 2],
            "code": ["bad", "X-2"],
            "numeric_code": [1, 2],
            "unknown_dtype": [1, 2],
        }
    )
    schema = {
        "bad_rule": "not-a-mapping",
        "nullable": {"nullable": False},
        "flag": {"dtype": "boolean"},
        "literal": {"allowed": "A"},
        "scalar_allowed": {"allowed": 1},
        "score": {"min": 0, "max": 10},
        "low_score": {"min": 0},
        "high_score": {"max": 10},
        "group": {"allowed": ["A", "B"]},
        "code": {"regex": r"X-\d"},
        "numeric_code": {"regex": r"\d"},
        "unknown_dtype": {"dtype": "imaginary"},
    }
    result = validate_dataframe(df, schema)
    assert result["ok"] is False
    assert any("must be a mapping" in err for err in result["errors"])
    assert any("contains null values" in err for err in result["errors"])
    assert any("does not match declared dtype" in err for err in result["errors"])
    assert any("unsupported dtype" in err for err in result["errors"])
    assert any("cannot use 'min'" in err for err in result["errors"])
    assert any("below the allowed minimum" in err for err in result["errors"])
    assert any("above the allowed maximum" in err for err in result["errors"])
    assert any("outside the allowed set" in err for err in result["errors"])
    assert any("cannot use 'regex'" in err for err in result["errors"])
    assert any("fail regex validation" in err for err in result["errors"])


def test_validate_dataframe_dtype_success_variants() -> None:
    df = pd.DataFrame(
        {
            "label": ["a", "b"],
            "category": pd.Series(["a", "b"], dtype="category"),
            "flag": [True, False],
            "when": pd.to_datetime(["2026-01-01", "2026-01-02"]),
        }
    )

    result = validate_dataframe(
        df,
        {
            "label": {"dtype": "string"},
            "category": {"dtype": "category"},
            "flag": {"dtype": "boolean"},
            "when": {"dtype": "datetime"},
        },
    )

    assert result["ok"] is True


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
