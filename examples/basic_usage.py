"""Minimal example usage for the template package."""

from __future__ import annotations

from design_research_analysis import build_default_blueprint, describe_project


def main() -> None:
    """Print the default project summary."""
    blueprint = build_default_blueprint("design-research-analysis")
    print(describe_project(blueprint))


if __name__ == "__main__":
    main()
