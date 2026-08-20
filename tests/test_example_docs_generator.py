"""Tests for recursive generated example documentation."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def _load_generator() -> ModuleType:
    """Load the generator directly from its script path."""
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "generate_example_docs.py"
    spec = importlib.util.spec_from_file_location("generate_example_docs", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_GENERATOR = _load_generator()


def test_nested_example_gets_safe_generated_page(tmp_path: Path) -> None:
    """Nested examples should be discovered, linked, and rendered without collisions."""
    example_path = tmp_path / "examples" / "nested" / "demo-example.py"
    example_path.parent.mkdir(parents=True)
    example_path.write_text(
        '"""Demo.\n\n'
        "## Introduction\nIntro.\n\n"
        "## Technical Implementation\nImplementation.\n\n"
        '## Expected Results\nResults.\n"""\n\n'
        "print('demo')\n",
        encoding="utf-8",
    )

    specs = _GENERATOR._build_specs(tmp_path)

    assert len(specs) == 1
    assert specs[0].rel_path == "examples/nested/demo-example.py"
    assert specs[0].slug == "nested__demo_example"
    rendered = _GENERATOR._render_example_page(specs[0])
    assert "../../examples/nested/demo-example.py" in rendered
    assert "PYTHONPATH=src python examples/nested/demo-example.py" in rendered


def test_stale_nested_generated_page_is_removed(tmp_path: Path) -> None:
    """Recursive cleanup should not leave orphaned generated pages behind."""
    docs_root = tmp_path / "docs" / "examples"
    nested_page = docs_root / "old" / "stale.rst"
    nested_page.parent.mkdir(parents=True)
    nested_page.write_text("stale\n", encoding="utf-8")
    root_index = docs_root / "index.rst"
    root_index.write_text("keep\n", encoding="utf-8")

    _GENERATOR._sync_stale_pages(
        generated_pages=set(),
        docs_examples_root=docs_root,
        check=False,
        stale=[],
    )

    assert not nested_page.exists()
    assert root_index.exists()
