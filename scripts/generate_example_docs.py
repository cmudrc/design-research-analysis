#!/usr/bin/env python3
"""Generate example Sphinx pages from top-level example docstrings."""

from __future__ import annotations

import argparse
import ast
import re
from dataclasses import dataclass
from pathlib import Path

REQUIRED_SECTIONS = (
    "Introduction",
    "Technical Implementation",
    "Expected Results",
)
OPTIONAL_SECTIONS = ("References",)
SUPPORTED_SECTIONS = REQUIRED_SECTIONS + OPTIONAL_SECTIONS

TITLE_TOKEN_OVERRIDES = {
    "api": "API",
    "pca": "PCA",
}


@dataclass(slots=True, frozen=True)
class ExampleDocSpec:
    """One example script plus parsed canonical docs sections."""

    rel_path: str
    slug: str
    title: str
    source_start_line: int
    sections: dict[str, str]


def _repo_root() -> Path:
    """Return repository root path."""
    return Path(__file__).resolve().parents[1]


def _discover_examples(repo_root: Path) -> list[Path]:
    """Discover runnable Python examples under ``examples/``."""
    examples_root = repo_root / "examples"
    discovered: list[Path] = []
    for path in sorted(examples_root.glob("*.py")):
        if path.name.startswith("_"):
            continue
        discovered.append(path)
    return discovered


def _parse_python_doc_text(path: Path) -> tuple[str, int]:
    """Parse module docstring text and source start line from one example."""
    source = path.read_text(encoding="utf-8")
    module = ast.parse(source, filename=str(path))
    docstring = ast.get_docstring(module, clean=False)
    if not isinstance(docstring, str) or not docstring.strip():
        raise ValueError(f"{path}: missing module docstring.")

    source_start_line = 1
    if module.body:
        first = module.body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
            and isinstance(first.end_lineno, int)
        ):
            source_start_line = first.end_lineno + 1

    lines = source.splitlines()
    while source_start_line <= len(lines) and not lines[source_start_line - 1].strip():
        source_start_line += 1

    return docstring, source_start_line


def _parse_canonical_sections(*, doc_text: str, source_path: Path) -> dict[str, str]:
    """Parse canonical docs sections from one module docstring."""
    heading_pattern = re.compile(r"^##\s+(.+?)\s*$")
    sections: dict[str, list[str]] = {}
    current_section: str | None = None

    for raw_line in doc_text.splitlines():
        line = raw_line.rstrip()
        match = heading_pattern.match(line.strip())
        if match is not None:
            heading = match.group(1).strip()
            if heading in SUPPORTED_SECTIONS:
                current_section = heading
                sections[current_section] = []
            else:
                current_section = None
            continue
        if current_section is not None:
            sections[current_section].append(line)

    missing = [section for section in REQUIRED_SECTIONS if section not in sections]
    if missing:
        raise ValueError(f"{source_path}: missing canonical section(s): {missing}")

    return {name: "\n".join(lines).strip() for name, lines in sections.items()}


def _slug_for_example(path: Path) -> str:
    """Build deterministic docs slug for one example path."""
    return path.stem.replace("-", "_")


def _title_for_example(path: Path) -> str:
    """Build human-readable page title for one example path."""
    label = path.stem.replace("-", " ").replace("_", " ")
    title_parts: list[str] = []
    for token in label.split(" "):
        normalized = token.strip().lower()
        if not normalized:
            continue
        title_parts.append(TITLE_TOKEN_OVERRIDES.get(normalized, normalized.capitalize()))
    return " ".join(title_parts)


def _render_optional_section(*, heading: str, body: str | None) -> list[str]:
    """Render one optional RST section block."""
    normalized = (body or "").strip()
    if not normalized:
        return []
    return [
        heading,
        "-" * len(heading),
        "",
        normalized,
        "",
    ]


def _render_example_page(spec: ExampleDocSpec) -> str:
    """Render one example page as RST."""
    run_command = f"PYTHONPATH=src python {spec.rel_path}"
    include_path = f"../../{spec.rel_path}"

    lines = [
        spec.title,
        "=" * len(spec.title),
        "",
        f"Source: ``{spec.rel_path}``",
        "",
        "Introduction",
        "------------",
        "",
        spec.sections["Introduction"],
        "",
        "Technical Implementation",
        "------------------------",
        "",
        spec.sections["Technical Implementation"],
        "",
        ".. literalinclude:: " + include_path,
        "   :language: python",
        f"   :lines: {spec.source_start_line}-",
        "   :linenos:",
        "",
        "Expected Results",
        "----------------",
        "",
        ".. rubric:: Run Command",
        "",
        ".. code-block:: bash",
        "",
        f"   {run_command}",
        "",
        spec.sections["Expected Results"],
        "",
    ]
    lines.extend(
        _render_optional_section(heading="References", body=spec.sections.get("References"))
    )
    return "\n".join(lines)


def _render_examples_index(specs: list[ExampleDocSpec]) -> str:
    """Render top-level examples index page as RST."""
    lines = [
        "Examples Guide",
        "==============",
        "",
        "Per-example documentation is generated from runnable example docstrings/comments.",
        "",
        ".. toctree::",
        "   :maxdepth: 1",
        "",
    ]
    for spec in specs:
        lines.append(f"   {spec.slug}")
    lines.append("")
    return "\n".join(lines)


def _build_specs(repo_root: Path) -> list[ExampleDocSpec]:
    """Build parsed docs specs for runnable examples."""
    specs: list[ExampleDocSpec] = []
    for path in _discover_examples(repo_root):
        doc_text, source_start_line = _parse_python_doc_text(path)
        sections = _parse_canonical_sections(doc_text=doc_text, source_path=path)
        rel_path = path.relative_to(repo_root).as_posix()
        specs.append(
            ExampleDocSpec(
                rel_path=rel_path,
                slug=_slug_for_example(path),
                title=_title_for_example(path),
                source_start_line=source_start_line,
                sections=sections,
            )
        )
    if not specs:
        raise ValueError("No examples found under examples/.")
    return sorted(specs, key=lambda item: item.rel_path)


def _sync_file(*, path: Path, content: str, check: bool, stale: list[str]) -> None:
    """Write one generated file or record drift in check mode."""
    desired = content.rstrip() + "\n"
    if path.exists():
        current = path.read_text(encoding="utf-8")
        if current == desired:
            return
    if check:
        stale.append(path.as_posix())
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(desired, encoding="utf-8")


def _sync_stale_pages(
    *,
    generated_pages: set[Path],
    docs_examples_root: Path,
    check: bool,
    stale: list[str],
) -> None:
    """Remove stale generated pages or report drift in check mode."""
    if not docs_examples_root.exists():
        return
    for existing in sorted(docs_examples_root.glob("*.rst")):
        if existing.name == "index.rst":
            continue
        if existing not in generated_pages:
            if check:
                stale.append(existing.as_posix())
            else:
                existing.unlink()


def generate(*, repo_root: Path, check: bool) -> int:
    """Generate docs pages or validate generated pages are up to date."""
    specs = _build_specs(repo_root)
    docs_examples_root = repo_root / "docs" / "examples"

    stale: list[str] = []
    generated_pages: set[Path] = set()

    _sync_file(
        path=docs_examples_root / "index.rst",
        content=_render_examples_index(specs),
        check=check,
        stale=stale,
    )

    for spec in specs:
        page_path = docs_examples_root / f"{spec.slug}.rst"
        generated_pages.add(page_path)
        _sync_file(
            path=page_path,
            content=_render_example_page(spec),
            check=check,
            stale=stale,
        )

    _sync_stale_pages(
        generated_pages=generated_pages,
        docs_examples_root=docs_examples_root,
        check=check,
        stale=stale,
    )

    if stale:
        print("Example docs are out of date:")
        for path in sorted(stale):
            print(f"- {path}")
        return 1

    if check:
        print("Example docs are up to date.")
    else:
        print("Generated example docs.")
    return 0


def main() -> int:
    """CLI entrypoint for example docs generation/check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="Validate generated docs are up to date."
    )
    args = parser.parse_args()
    return generate(repo_root=_repo_root(), check=args.check)


if __name__ == "__main__":
    raise SystemExit(main())
