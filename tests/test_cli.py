from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import design_research_analysis.cli as cli_module
from design_research_analysis.cli import main


def _write_fixture_csv(path: Path) -> None:
    rows = [
        {
            "timestamp": "2026-01-01T10:00:00Z",
            "session_id": "s1",
            "event_type": "A",
            "text": "good clear result",
            "record_id": "r1",
            "value": "1.0",
            "group": "g1",
            "x1": "1.0",
            "x2": "0.0",
            "y": "2.0",
        },
        {
            "timestamp": "2026-01-01T10:00:01Z",
            "session_id": "s1",
            "event_type": "B",
            "text": "bad unclear problem",
            "record_id": "r2",
            "value": "2.0",
            "group": "g1",
            "x1": "2.0",
            "x2": "1.0",
            "y": "4.0",
        },
        {
            "timestamp": "2026-01-01T10:00:02Z",
            "session_id": "s2",
            "event_type": "A",
            "text": "great collaborative success",
            "record_id": "r3",
            "value": "5.0",
            "group": "g2",
            "x1": "3.0",
            "x2": "1.0",
            "y": "6.0",
        },
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class _FakeResult:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def to_dict(self) -> dict[str, Any]:
        return dict(self._payload)


class _FakeConvergence:
    def __init__(self) -> None:
        self.distance_trajectories = {"s1": [0.3, 0.1], "s2": [0.2]}

    def to_dict(self) -> dict[str, Any]:
        return {"direction_by_group": {"s1": "converging", "s2": "stable"}}


class _FakeFigure:
    def __init__(self) -> None:
        self.saved: list[str] = []
        self.cleared = False

    def savefig(self, path: str, **_: object) -> None:
        self.saved.append(path)

    def clf(self) -> None:
        self.cleared = True


class _FakeCodebook:
    def __init__(self) -> None:
        self.shape = (2, 3)
        self.columns = ["column", "inferred_dtype", "description"]
        self.written_path: str | None = None

    def to_csv(self, path: Path, index: bool = False) -> None:
        _ = index
        self.written_path = str(path)
        Path(path).write_text(
            "column,inferred_dtype,description\nfirst,integer,\nsecond,string,Condition\n",
            encoding="utf-8",
        )


def _assert_envelope(payload: dict[str, Any], *, analysis: str, mode: str) -> None:
    assert payload["analysis"] == analysis
    assert payload["mode"] == mode
    assert payload["output_schema_version"] == "1.0"


def test_cli_validate_table_smoke(tmp_path: Path) -> None:
    input_csv = tmp_path / "input.csv"
    summary_json = tmp_path / "validate.json"
    _write_fixture_csv(input_csv)

    exit_code = main(
        [
            "validate-table",
            "--input",
            str(input_csv),
            "--summary-json",
            str(summary_json),
        ]
    )

    assert exit_code == 0
    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    assert payload["is_valid"] is True
    _assert_envelope(payload, analysis="table", mode="validate")


def test_cli_run_sequence_markov_smoke(tmp_path: Path) -> None:
    input_csv = tmp_path / "input.csv"
    summary_json = tmp_path / "sequence.json"
    _write_fixture_csv(input_csv)

    exit_code = main(
        [
            "run-sequence",
            "--input",
            str(input_csv),
            "--summary-json",
            str(summary_json),
            "--mode",
            "markov",
        ]
    )

    assert exit_code == 0
    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    _assert_envelope(payload, analysis="sequence", mode="markov")
    assert "transition_matrix" in payload["result"]


def test_cli_run_stats_regression_smoke(tmp_path: Path) -> None:
    input_csv = tmp_path / "input.csv"
    summary_json = tmp_path / "stats.json"
    _write_fixture_csv(input_csv)

    exit_code = main(
        [
            "run-stats",
            "--input",
            str(input_csv),
            "--summary-json",
            str(summary_json),
            "--mode",
            "regression",
            "--x-columns",
            "x1,x2",
            "--y-column",
            "y",
        ]
    )

    assert exit_code == 0
    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    _assert_envelope(payload, analysis="stats", mode="regression")
    assert "coefficients" in payload["result"]


def test_cli_helpers_cover_json_and_csv_serialization(tmp_path: Path) -> None:
    json_path = tmp_path / "rows.json"
    json_path.write_text(
        json.dumps([{"timestamp": "2026-01-01T10:00:00Z", "text": "hello"}]),
        encoding="utf-8",
    )
    rows = cli_module._load_table(str(json_path))
    assert rows[0]["text"] == "hello"

    serialized = cli_module._serialize_for_json(
        {
            "when": datetime(2026, 1, 1, 10, 0, 0),
            "vector": np.asarray([1.0, 2.0]),
            "items": ("a", "b"),
        }
    )
    assert isinstance(serialized["when"], str)
    assert serialized["vector"] == [1.0, 2.0]
    assert serialized["items"] == ["a", "b"]

    csv_path = tmp_path / "empty.csv"
    cli_module._write_csv(str(csv_path), [])
    assert csv_path.read_text(encoding="utf-8") == ""


def test_cli_load_mapper_validation_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    assert cli_module._load_mapper(None) is None
    with pytest.raises(ValueError, match="module:function"):
        cli_module._load_mapper("bad")

    fake_module = SimpleNamespace(not_callable=123)
    monkeypatch.setitem(sys.modules, "fake_mapper_mod", fake_module)
    with pytest.raises(ValueError, match="did not resolve"):
        cli_module._load_mapper("fake_mapper_mod:not_callable")


def test_cli_load_mapper_accepts_dotted_path(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_mapper = lambda row: row  # noqa: E731
    fake_module = SimpleNamespace(mapper=fake_mapper)
    monkeypatch.setitem(sys.modules, "fake_mapper_pkg", fake_module)

    loaded = cli_module._load_mapper("fake_mapper_pkg.mapper")
    assert loaded is fake_mapper


def test_cli_run_language_topic_error_and_trajectory_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_csv = tmp_path / "input.csv"
    summary_json = tmp_path / "language.json"
    trajectory_csv = tmp_path / "trajectory.csv"
    _write_fixture_csv(input_csv)

    monkeypatch.setattr(
        cli_module,
        "compute_language_convergence",
        lambda *args, **kwargs: _FakeConvergence(),
    )
    monkeypatch.setattr(cli_module, "score_sentiment", lambda *args, **kwargs: {"mean_score": 0.1})

    def _raise_topic(*args: object, **kwargs: object) -> dict[str, Any]:
        raise ImportError("topic deps missing")

    monkeypatch.setattr(cli_module, "fit_topic_model", _raise_topic)

    exit_code = main(
        [
            "run-language",
            "--input",
            str(input_csv),
            "--summary-json",
            str(summary_json),
            "--trajectory-csv",
            str(trajectory_csv),
            "--include-topics",
        ]
    )

    assert exit_code == 0
    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    assert "topic_model_error" in payload
    _assert_envelope(payload, analysis="language", mode="language")
    assert trajectory_csv.read_text(encoding="utf-8").startswith("group,step,semantic_distance")


def test_cli_run_dimred_with_stubbed_backends(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_csv = tmp_path / "input.csv"
    summary_json = tmp_path / "dimred.json"
    projection_csv = tmp_path / "projection.csv"
    _write_fixture_csv(input_csv)

    fake_embeddings = SimpleNamespace(
        embeddings=np.asarray([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]),
        record_ids=["r1", "r2", "r3"],
        to_dict=lambda: {"shape": [3, 2]},
    )
    fake_projection = SimpleNamespace(
        projection=np.asarray([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),
        to_dict=lambda: {"shape": [3, 2]},
    )
    monkeypatch.setattr(cli_module, "embed_records", lambda *args, **kwargs: fake_embeddings)
    monkeypatch.setattr(cli_module, "reduce_dimensions", lambda *args, **kwargs: fake_projection)
    monkeypatch.setattr(
        cli_module,
        "cluster_projection",
        lambda *args, **kwargs: {"labels": [0, 1, 0], "method": "kmeans"},
    )
    monkeypatch.setattr(
        cli_module,
        "compute_design_space_coverage",
        lambda *args, **kwargs: {"pairwise_spread": {"mean": 0.42}},
    )
    monkeypatch.setattr(
        cli_module,
        "compute_idea_space_trajectory",
        lambda *args, **kwargs: {"groups": {"s1": {"path_length": 1.0}}},
    )
    monkeypatch.setattr(
        cli_module,
        "compute_divergence_convergence",
        lambda *args, **kwargs: {"groups": {"s1": {"dominant_direction": "diverging"}}},
    )

    exit_code = main(
        [
            "run-dimred",
            "--input",
            str(input_csv),
            "--summary-json",
            str(summary_json),
            "--projection-csv",
            str(projection_csv),
        ]
    )

    assert exit_code == 0
    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    _assert_envelope(payload, analysis="dimred", mode="pca")
    assert payload["coverage"]["pairwise_spread"]["mean"] == 0.42
    assert payload["trajectory"]["divergence_convergence"]["groups"]["s1"][
        "dominant_direction"
    ] == ("diverging")
    assert projection_csv.exists()


def test_cli_run_sequence_discrete_and_text_modes_with_matrix_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_csv = tmp_path / "input.csv"
    _write_fixture_csv(input_csv)

    fake_figure = _FakeFigure()
    monkeypatch.setattr(
        cli_module,
        "plot_transition_matrix",
        lambda *args, **kwargs: (fake_figure, object()),
    )
    monkeypatch.setattr(
        cli_module,
        "fit_discrete_hmm_from_table",
        lambda *args, **kwargs: _FakeResult({"transmat": [[0.5, 0.5], [0.4, 0.6]]}),
    )
    monkeypatch.setattr(
        cli_module,
        "fit_text_gaussian_hmm_from_table",
        lambda *args, **kwargs: _FakeResult({"means": [[0.0], [1.0]]}),
    )

    seq_json = tmp_path / "discrete.json"
    png_path = tmp_path / "matrix.png"
    exit_discrete = main(
        [
            "run-sequence",
            "--input",
            str(input_csv),
            "--summary-json",
            str(seq_json),
            "--mode",
            "discrete-hmm",
            "--matrix-png",
            str(png_path),
        ]
    )
    assert exit_discrete == 0
    assert fake_figure.saved
    discrete_payload = json.loads(seq_json.read_text(encoding="utf-8"))
    _assert_envelope(discrete_payload, analysis="sequence", mode="discrete-hmm")

    text_json = tmp_path / "text.json"
    exit_text = main(
        [
            "run-sequence",
            "--input",
            str(input_csv),
            "--summary-json",
            str(text_json),
            "--mode",
            "text-gaussian-hmm",
        ]
    )
    assert exit_text == 0
    payload = json.loads(text_json.read_text(encoding="utf-8"))
    _assert_envelope(payload, analysis="sequence", mode="text-gaussian-hmm")


def test_cli_run_stats_compare_and_mixed_modes_with_stubs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_csv = tmp_path / "input.csv"
    _write_fixture_csv(input_csv)

    monkeypatch.setattr(
        cli_module,
        "compare_groups",
        lambda *args, **kwargs: _FakeResult({"ok": True}),
    )
    monkeypatch.setattr(
        cli_module,
        "fit_mixed_effects",
        lambda *args, **kwargs: _FakeResult({"success": True}),
    )

    compare_json = tmp_path / "compare.json"
    exit_compare = main(
        [
            "run-stats",
            "--input",
            str(input_csv),
            "--summary-json",
            str(compare_json),
            "--mode",
            "compare",
        ]
    )
    assert exit_compare == 0
    compare_payload = json.loads(compare_json.read_text(encoding="utf-8"))
    _assert_envelope(compare_payload, analysis="stats", mode="compare")

    mixed_json = tmp_path / "mixed.json"
    exit_mixed = main(
        [
            "run-stats",
            "--input",
            str(input_csv),
            "--summary-json",
            str(mixed_json),
            "--mode",
            "mixed",
        ]
    )
    assert exit_mixed == 0
    mixed_payload = json.loads(mixed_json.read_text(encoding="utf-8"))
    _assert_envelope(mixed_payload, analysis="stats", mode="mixed")


def test_cli_invalid_extension_and_missing_x_columns(tmp_path: Path) -> None:
    bad_input = tmp_path / "input.bad"
    bad_input.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported input format"):
        cli_module._load_table(str(bad_input))

    csv_input = tmp_path / "input.csv"
    _write_fixture_csv(csv_input)
    with pytest.raises(ValueError, match="requires --x-columns"):
        main(
            [
                "run-stats",
                "--input",
                str(csv_input),
                "--summary-json",
                str(tmp_path / "stats.json"),
                "--mode",
                "regression",
                "--x-columns",
                "",
            ]
        )


def test_cli_profile_dataset_and_validate_dataset_modes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli_module, "_load_dataframe", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        cli_module,
        "profile_dataframe",
        lambda *args, **kwargs: {"n_rows": 3, "warnings": []},
    )

    profile_json = tmp_path / "profile.json"
    exit_profile = main(
        [
            "profile-dataset",
            "--input",
            str(tmp_path / "data.csv"),
            "--summary-json",
            str(profile_json),
        ]
    )
    assert exit_profile == 0
    profile_payload = json.loads(profile_json.read_text(encoding="utf-8"))
    _assert_envelope(profile_payload, analysis="dataset", mode="profile")
    assert profile_payload["result"]["n_rows"] == 3

    monkeypatch.setattr(
        cli_module,
        "validate_dataframe",
        lambda *args, **kwargs: {"ok": False, "errors": ["bad"], "warnings": [], "summary": {}},
    )
    validate_json = tmp_path / "validate_dataset.json"
    exit_validate = main(
        [
            "validate-dataset",
            "--input",
            str(tmp_path / "data.csv"),
            "--summary-json",
            str(validate_json),
            "--schema-json",
            '{"participant_id": {"unique": true}}',
        ]
    )
    assert exit_validate == 2
    validate_payload = json.loads(validate_json.read_text(encoding="utf-8"))
    _assert_envelope(validate_payload, analysis="dataset", mode="validate")
    assert validate_payload["result"]["ok"] is False


def test_cli_validate_dataset_requires_schema_source(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="schema is required"):
        main(
            [
                "validate-dataset",
                "--input",
                str(tmp_path / "data.csv"),
                "--summary-json",
                str(tmp_path / "summary.json"),
            ]
        )


def test_cli_generate_codebook_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_codebook = _FakeCodebook()
    monkeypatch.setattr(cli_module, "_load_dataframe", lambda *args, **kwargs: object())
    monkeypatch.setattr(cli_module, "generate_codebook", lambda *args, **kwargs: fake_codebook)

    summary_json = tmp_path / "codebook_summary.json"
    codebook_csv = tmp_path / "codebook.csv"
    exit_code = main(
        [
            "generate-codebook",
            "--input",
            str(tmp_path / "data.csv"),
            "--summary-json",
            str(summary_json),
            "--codebook-csv",
            str(codebook_csv),
            "--descriptions-json",
            '{"second": "Condition"}',
        ]
    )
    assert exit_code == 0
    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    _assert_envelope(payload, analysis="dataset", mode="codebook")
    assert payload["result"]["codebook_csv"] == str(codebook_csv)
    assert codebook_csv.exists()


def test_cli_capture_context_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def _fake_capture(*args: object, **kwargs: object) -> dict[str, Any]:
        captured["capture_kwargs"] = dict(kwargs)
        return {"random_seed": kwargs.get("seed"), "inputs": []}

    def _fake_manifest(context: dict[str, Any], outpath: str) -> Path:
        captured["manifest_context"] = dict(context)
        out = Path(outpath)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("{}", encoding="utf-8")
        return out

    monkeypatch.setattr(cli_module, "capture_run_context", _fake_capture)
    monkeypatch.setattr(cli_module, "write_run_manifest", _fake_manifest)

    summary_json = tmp_path / "context_summary.json"
    manifest_json = tmp_path / "run_manifest.json"
    exit_code = main(
        [
            "capture-context",
            "--summary-json",
            str(summary_json),
            "--manifest-json",
            str(manifest_json),
            "--seed",
            "7",
            "--input-path",
            str(tmp_path / "input.csv"),
            "--extra-json",
            '{"stage": "test"}',
        ]
    )
    assert exit_code == 0
    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    _assert_envelope(payload, analysis="runtime", mode="capture-context")
    assert payload["result"]["manifest_json"] == str(manifest_json)
    assert manifest_json.exists()
    assert captured["capture_kwargs"]["extra"] == {"stage": "test"}
