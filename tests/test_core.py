"""Tests for the core template helpers."""

from __future__ import annotations

from design_research_analysis.core import (
    ProjectBlueprint,
    build_default_blueprint,
    describe_project,
    normalize_package_name,
)


def test_normalize_package_name_rewrites_non_identifier_tokens() -> None:
    """Normalize a repository name into an import-safe package token."""

    assert normalize_package_name("Design Research Template") == "design_research_template"


def test_build_default_blueprint_uses_normalized_package_name() -> None:
    """Default blueprints should derive the import package name."""

    blueprint = build_default_blueprint("design-research-analysis")

    assert blueprint == ProjectBlueprint(
        name="design-research-analysis",
        package_name="design_research_analysis",
    )


def test_describe_project_includes_expected_summary_fields() -> None:
    """The rendered project summary should include the major template defaults."""

    blueprint = build_default_blueprint("design-research-analysis")
    description = describe_project(blueprint)

    assert "Project: design-research-analysis" in description
    assert "Import package: design_research_analysis" in description
    assert "Toolchain: ruff, mypy, pytest, sphinx" in description
