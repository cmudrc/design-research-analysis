"""Tests for lightweight docs consistency helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def _load_checker() -> ModuleType:
    """Load the checker directly from its script path."""
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "check_docs_consistency.py"
    spec = importlib.util.spec_from_file_location("check_docs_consistency", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_CHECKER = _load_checker()
discover_example_paths = _CHECKER.discover_example_paths
extract_documented_api_names = _CHECKER.extract_documented_api_names
extract_documented_example_paths = _CHECKER.extract_documented_example_paths
extract_public_exports = _CHECKER.extract_public_exports
extract_toctree_entries = _CHECKER.extract_toctree_entries
find_api_inventory_differences = _CHECKER.find_api_inventory_differences
find_example_inventory_differences = _CHECKER.find_example_inventory_differences


def test_extract_toctree_entries_skips_external_links(tmp_path: Path) -> None:
    index_path = tmp_path / "index.rst"
    index_path.write_text(
        "\n".join(
            [
                ".. toctree::",
                "   quickstart",
                "   Contributing <https://example.com/CONTRIBUTING.md>",
                "",
                ".. toctree::",
                "   Reference <reference/index>",
            ]
        ),
        encoding="utf-8",
    )

    assert extract_toctree_entries(index_path) == ("quickstart", "reference/index")


def test_extract_public_exports_reads_literal_all(tmp_path: Path) -> None:
    init_path = tmp_path / "__init__.py"
    init_path.write_text('__all__ = ["Result", "analyze"]\n', encoding="utf-8")

    assert extract_public_exports(init_path) == ("Result", "analyze")


def test_extract_documented_api_names_reads_only_inventory_block(tmp_path: Path) -> None:
    """Incidental prose outside the grouped inventory should not count."""
    api_path = tmp_path / "api.rst"
    api_path.write_text(
        "The package exposes ``Result``.\n\n"
        "Top-level groups:\n\n"
        "- Results: ``analyze``\n"
        "  ``summarize``\n\n"
        "Comparison helpers include ``difference``.\n",
        encoding="utf-8",
    )

    assert extract_documented_api_names(api_path) == {"analyze", "summarize"}


def test_find_api_inventory_differences_reports_missing_and_stale_names(
    tmp_path: Path,
) -> None:
    """API comparison should reject both omitted exports and stale inventory rows."""
    init_path = tmp_path / "__init__.py"
    init_path.write_text('__all__ = ["Result", "analyze"]\n', encoding="utf-8")
    api_path = tmp_path / "api.rst"
    api_path.write_text(
        "Top-level groups:\n\n- Results: ``Result``, ``removed_export``\n",
        encoding="utf-8",
    )

    assert find_api_inventory_differences(init_path, api_path) == (
        ("analyze",),
        ("removed_export",),
    )


def test_example_inventory_is_exact_recursive_and_symmetric(tmp_path: Path) -> None:
    """Example comparison should use exact relative paths in both directions."""
    examples_dir = tmp_path / "examples"
    examples_dir.mkdir()
    (examples_dir / "listed.py").write_text("", encoding="utf-8")
    nested_dir = examples_dir / "nested"
    nested_dir.mkdir()
    (nested_dir / "missing.py").write_text("", encoding="utf-8")
    (nested_dir / "_helper.py").write_text("", encoding="utf-8")
    readme_path = examples_dir / "README.md"
    readme_path.write_text(
        "- `listed.py`: listed example\n"
        "The string `nested/missing.py.old` is not an inventory entry.\n"
        "- `stale.py`: removed example\n",
        encoding="utf-8",
    )

    assert discover_example_paths(examples_dir) == {"listed.py", "nested/missing.py"}
    assert extract_documented_example_paths(readme_path) == {"listed.py", "stale.py"}
    assert find_example_inventory_differences(examples_dir, readme_path) == (
        ("nested/missing.py",),
        ("stale.py",),
    )
