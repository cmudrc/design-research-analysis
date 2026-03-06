"""Thin command-line interface for design research analysis pipelines."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .dimred import cluster_projection, embed_records, reduce_dimensions
from .language import compute_language_convergence, fit_topic_model, score_sentiment
from .sequence import (
    fit_discrete_hmm_from_table,
    fit_markov_chain_from_table,
    fit_text_gaussian_hmm_from_table,
    plot_transition_matrix,
)
from .stats import compare_groups, fit_mixed_effects, fit_regression
from .table import UnifiedTableConfig, coerce_unified_table, validate_unified_table

_OUTPUT_SCHEMA_VERSION = "1.0"


def _load_table(path: str) -> list[dict[str, Any]]:
    input_path = Path(path)
    suffix = input_path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        return coerce_unified_table(payload)
    if suffix in {".csv", ".tsv"}:
        delimiter = "," if suffix == ".csv" else "\t"
        with input_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            return coerce_unified_table(list(reader))
    raise ValueError("Unsupported input format. Use .csv, .tsv, or .json.")


def _serialize_for_json(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _serialize_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_serialize_for_json(item) for item in value]
    if isinstance(value, tuple):
        return [_serialize_for_json(item) for item in value]
    return value


def _write_json(path: str, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(_serialize_for_json(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _write_csv(path: str, rows: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _base_payload(*, analysis: str, mode: str) -> dict[str, Any]:
    return {
        "analysis": analysis,
        "mode": mode,
        "output_schema_version": _OUTPUT_SCHEMA_VERSION,
    }


def _load_mapper(spec: str | None) -> Any:
    if spec is None:
        return None
    if ":" in spec:
        module_name, func_name = spec.split(":", 1)
    elif "." in spec:
        module_name, func_name = spec.rsplit(".", 1)
    else:
        raise ValueError("Mapper spec must use 'module:function' or 'module.function' format.")
    module = importlib.import_module(module_name)
    mapper = getattr(module, func_name, None)
    if mapper is None or not callable(mapper):
        raise ValueError(f"Mapper '{spec}' did not resolve to a callable.")
    return mapper


def _cmd_validate_table(args: argparse.Namespace) -> int:
    rows = _load_table(args.input)
    config = UnifiedTableConfig()
    report = validate_unified_table(rows, config=config)
    payload = _base_payload(analysis="table", mode="validate")
    payload.update(report.to_dict())
    _write_json(args.summary_json, payload)
    return 0 if report.is_valid else 2


def _cmd_run_language(args: argparse.Namespace) -> int:
    rows = _load_table(args.input)
    convergence = compute_language_convergence(
        rows,
        text_column=args.text_column,
        group_column=args.group_column,
        window_size=args.window_size,
    )
    sentiment = score_sentiment(rows, text_column=args.text_column)

    payload: dict[str, Any] = {
        **_base_payload(analysis="language", mode="language"),
        "convergence": convergence.to_dict(),
        "sentiment": sentiment,
    }

    if args.include_topics:
        try:
            topic = fit_topic_model(
                rows,
                text_column=args.text_column,
                n_topics=args.n_topics,
                random_state=args.seed,
            )
            payload["topic_model"] = topic
        except ImportError as exc:
            payload["topic_model_error"] = str(exc)

    _write_json(args.summary_json, payload)

    trajectory_rows: list[dict[str, Any]] = []
    for group, distances in convergence.distance_trajectories.items():
        for index, distance in enumerate(distances):
            trajectory_rows.append(
                {
                    "group": group,
                    "step": index,
                    "semantic_distance": float(distance),
                }
            )
    if args.trajectory_csv:
        _write_csv(args.trajectory_csv, trajectory_rows)
    return 0


def _cmd_run_dimred(args: argparse.Namespace) -> int:
    rows = _load_table(args.input)
    embeddings = embed_records(
        rows,
        text_column=args.text_column,
        record_id_column=args.record_id_column,
    )
    projection = reduce_dimensions(
        embeddings.embeddings,
        method=args.method,
        n_components=args.n_components,
        random_state=args.seed,
        perplexity=args.perplexity,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
    )
    clustering = cluster_projection(
        projection.projection,
        method=args.cluster_method,
        n_clusters=args.n_clusters,
        random_state=args.seed,
    )

    projection_rows: list[dict[str, Any]] = []
    for idx, record_id in enumerate(embeddings.record_ids):
        row: dict[str, Any] = {"record_id": record_id}
        for component in range(projection.projection.shape[1]):
            row[f"component_{component + 1}"] = float(projection.projection[idx, component])
        row["cluster_label"] = int(clustering["labels"][idx])
        projection_rows.append(row)

    _write_json(
        args.summary_json,
        {
            **_base_payload(analysis="dimred", mode=args.method),
            "embedding": embeddings.to_dict(),
            "projection": projection.to_dict(),
            "clustering": clustering,
        },
    )
    _write_csv(args.projection_csv, projection_rows)
    return 0


def _cmd_run_sequence(args: argparse.Namespace) -> int:
    rows = _load_table(args.input)
    actor_mapper = _load_mapper(args.actor_mapper)
    event_mapper = _load_mapper(args.event_mapper)
    session_mapper = _load_mapper(args.session_mapper)
    text_mapper = _load_mapper(args.text_mapper)

    if args.mode == "markov":
        markov_result = fit_markov_chain_from_table(
            rows,
            order=args.order,
            smoothing=args.smoothing,
            event_column=args.event_column,
            session_column=args.session_column,
            actor_column=args.actor_column,
            include_actor_in_token=args.include_actor_in_token,
            actor_mapper=actor_mapper,
            event_mapper=event_mapper,
            session_mapper=session_mapper,
        )
        payload = {
            **_base_payload(analysis="sequence", mode="markov"),
            "result": markov_result.to_dict(),
        }
        if args.matrix_png:
            figure, _ = plot_transition_matrix(markov_result, annotate=False)
            figure.savefig(args.matrix_png, dpi=150, bbox_inches="tight")
            figure.clf()
    elif args.mode == "discrete-hmm":
        discrete_result = fit_discrete_hmm_from_table(
            rows,
            n_states=args.n_states,
            n_iter=args.n_iter,
            seed=args.seed,
            backend=args.backend,
            event_column=args.event_column,
            session_column=args.session_column,
            actor_column=args.actor_column,
            include_actor_in_token=args.include_actor_in_token,
            actor_mapper=actor_mapper,
            event_mapper=event_mapper,
            session_mapper=session_mapper,
        )
        payload = {
            **_base_payload(analysis="sequence", mode="discrete-hmm"),
            "result": discrete_result.to_dict(),
        }
        if args.matrix_png:
            figure, _ = plot_transition_matrix(discrete_result, annotate=False)
            figure.savefig(args.matrix_png, dpi=150, bbox_inches="tight")
            figure.clf()
    else:
        gaussian_result = fit_text_gaussian_hmm_from_table(
            rows,
            text_column=args.text_column,
            session_column=args.session_column,
            n_states=args.n_states,
            model_name=args.model_name,
            normalize=args.normalize_embeddings,
            batch_size=args.batch_size,
            device=args.device,
            n_iter=args.n_iter,
            seed=args.seed,
            backend=args.backend,
            session_mapper=session_mapper,
            text_mapper=text_mapper,
        )
        payload = {
            **_base_payload(analysis="sequence", mode="text-gaussian-hmm"),
            "result": gaussian_result.to_dict(),
        }
        if args.matrix_png:
            figure, _ = plot_transition_matrix(gaussian_result, annotate=False)
            figure.savefig(args.matrix_png, dpi=150, bbox_inches="tight")
            figure.clf()

    _write_json(args.summary_json, payload)
    return 0


def _cmd_run_stats(args: argparse.Namespace) -> int:
    rows = _load_table(args.input)

    if args.mode == "compare":
        compare_result = compare_groups(
            data=rows,
            value_column=args.value_column,
            group_column=args.group_column,
            method=args.method,
        )
        payload = {
            **_base_payload(analysis="stats", mode="compare"),
            "result": compare_result.to_dict(),
        }
    elif args.mode == "regression":
        x_columns = [column.strip() for column in args.x_columns.split(",") if column.strip()]
        if not x_columns:
            raise ValueError("regression mode requires --x-columns.")

        X_rows: list[list[float]] = []
        y_values: list[float] = []
        for index, row in enumerate(rows):
            try:
                X_rows.append([float(row[column]) for column in x_columns])
                y_values.append(float(row[args.y_column]))
            except KeyError as exc:
                raise ValueError(f"Row {index} is missing regression column: {exc}") from exc
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Row {index} contains non-numeric regression values.") from exc

        regression_result = fit_regression(
            X_rows,
            y_values,
            feature_names=x_columns,
            add_intercept=True,
        )
        payload = {
            **_base_payload(analysis="stats", mode="regression"),
            "result": regression_result.to_dict(),
        }
    else:
        mixed_result = fit_mixed_effects(
            rows,
            formula=args.formula,
            group_column=args.group_column,
            backend="statsmodels",
            reml=args.reml,
            max_iter=args.max_iter,
        )
        payload = {
            **_base_payload(analysis="stats", mode="mixed"),
            "result": mixed_result.to_dict(),
        }

    _write_json(args.summary_json, payload)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="design-research-analysis")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate-table", help="Validate unified table input.")
    validate_parser.add_argument("--input", required=True)
    validate_parser.add_argument("--summary-json", required=True)
    validate_parser.set_defaults(func=_cmd_validate_table)

    language_parser = subparsers.add_parser("run-language", help="Run language analyses.")
    language_parser.add_argument("--input", required=True)
    language_parser.add_argument("--summary-json", required=True)
    language_parser.add_argument("--trajectory-csv")
    language_parser.add_argument("--text-column", default="text")
    language_parser.add_argument("--group-column", default="session_id")
    language_parser.add_argument("--window-size", type=int, default=3)
    language_parser.add_argument("--seed", type=int, default=0)
    language_parser.add_argument("--include-topics", action="store_true")
    language_parser.add_argument("--n-topics", type=int, default=5)
    language_parser.set_defaults(func=_cmd_run_language)

    dimred_parser = subparsers.add_parser(
        "run-dimred",
        help="Run embedding and dim-red analyses.",
    )
    dimred_parser.add_argument("--input", required=True)
    dimred_parser.add_argument("--summary-json", required=True)
    dimred_parser.add_argument("--projection-csv", required=True)
    dimred_parser.add_argument("--text-column", default="text")
    dimred_parser.add_argument("--record-id-column", default="record_id")
    dimred_parser.add_argument("--method", choices=["pca", "tsne", "umap"], default="pca")
    dimred_parser.add_argument("--n-components", type=int, default=2)
    dimred_parser.add_argument("--seed", type=int, default=0)
    dimred_parser.add_argument("--perplexity", type=float, default=30.0)
    dimred_parser.add_argument("--n-neighbors", type=int, default=15)
    dimred_parser.add_argument("--min-dist", type=float, default=0.1)
    dimred_parser.add_argument(
        "--cluster-method",
        choices=["kmeans", "agglomerative"],
        default="kmeans",
    )
    dimred_parser.add_argument("--n-clusters", type=int, default=3)
    dimred_parser.set_defaults(func=_cmd_run_dimred)

    sequence_parser = subparsers.add_parser("run-sequence", help="Run sequence-model analyses.")
    sequence_parser.add_argument("--input", required=True)
    sequence_parser.add_argument("--summary-json", required=True)
    sequence_parser.add_argument(
        "--mode",
        choices=["markov", "discrete-hmm", "text-gaussian-hmm"],
        default="markov",
    )
    sequence_parser.add_argument("--event-column", default="event_type")
    sequence_parser.add_argument("--session-column", default="session_id")
    sequence_parser.add_argument("--actor-column", default="actor_id")
    sequence_parser.add_argument("--text-column", default="text")
    sequence_parser.add_argument("--actor-mapper")
    sequence_parser.add_argument("--event-mapper")
    sequence_parser.add_argument("--session-mapper")
    sequence_parser.add_argument("--text-mapper")
    sequence_parser.add_argument("--include-actor-in-token", action="store_true")
    sequence_parser.add_argument("--order", type=int, default=1)
    sequence_parser.add_argument("--smoothing", type=float, default=1.0)
    sequence_parser.add_argument("--n-states", type=int, default=3)
    sequence_parser.add_argument("--n-iter", type=int, default=100)
    sequence_parser.add_argument("--seed", type=int, default=0)
    sequence_parser.add_argument("--backend", default="hmmlearn")
    sequence_parser.add_argument("--model-name", default="all-MiniLM-L6-v2")
    normalize_group = sequence_parser.add_mutually_exclusive_group()
    normalize_group.add_argument(
        "--normalize-embeddings",
        action="store_true",
        dest="normalize_embeddings",
        default=True,
    )
    normalize_group.add_argument(
        "--no-normalize-embeddings",
        action="store_false",
        dest="normalize_embeddings",
    )
    sequence_parser.add_argument("--batch-size", type=int, default=32)
    sequence_parser.add_argument("--device", default="auto")
    sequence_parser.add_argument("--matrix-png")
    sequence_parser.set_defaults(func=_cmd_run_sequence)

    stats_parser = subparsers.add_parser("run-stats", help="Run statistical analyses.")
    stats_parser.add_argument("--input", required=True)
    stats_parser.add_argument("--summary-json", required=True)
    stats_parser.add_argument(
        "--mode",
        choices=["compare", "regression", "mixed"],
        default="compare",
    )
    stats_parser.add_argument("--value-column", default="value")
    stats_parser.add_argument("--group-column", default="group")
    stats_parser.add_argument("--method", default="auto")
    stats_parser.add_argument("--x-columns", default="")
    stats_parser.add_argument("--y-column", default="y")
    stats_parser.add_argument("--formula", default="y ~ x")
    stats_parser.add_argument("--reml", action="store_true")
    stats_parser.add_argument("--max-iter", type=int, default=200)
    stats_parser.set_defaults(func=_cmd_run_stats)

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
