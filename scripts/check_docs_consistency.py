"""Run a few lightweight consistency checks for the docs tree."""

from __future__ import annotations

import re
from pathlib import Path

DOCS_DIR = Path("docs")
INDEX_PATH = DOCS_DIR / "index.rst"
API_PATH = DOCS_DIR / "api.rst"
README_PATH = Path("README.md")
_LABELED_TARGET_RE = re.compile(r"^.+<([^>]+)>$")


def _normalize_toctree_target(entry: str) -> str | None:
    """Normalize one toctree entry into a local docs target, when applicable.

    Args:
        entry: Raw stripped toctree line.

    Returns:
        The local target path without a ``.rst`` suffix, or ``None`` for
        external links.
    """
    match = _LABELED_TARGET_RE.match(entry)
    target = match.group(1).strip() if match else entry.strip()
    if "://" in target:
        return None
    if target.endswith(".rst"):
        return target[:-4]
    return target


def extract_toctree_entries(index_path: Path) -> tuple[str, ...]:
    """Extract local document entries from all toctrees in ``index.rst``.

    Args:
        index_path: Path to the docs index file.

    Returns:
        The referenced local document names without suffixes.
    """
    entries: list[str] = []
    in_toctree = False
    for line in index_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped == ".. toctree::":
            in_toctree = True
            continue
        if not in_toctree:
            continue
        if not stripped:
            continue
        if stripped.startswith(":"):
            continue
        if line.startswith("   "):
            target = _normalize_toctree_target(stripped)
            if target is not None:
                entries.append(target)
            continue
        in_toctree = False
    return tuple(entries)


def validate_docs_tree() -> list[str]:
    """Collect any missing or inconsistent documentation references.

    Returns:
        A list of validation error messages.
    """
    errors: list[str] = []
    if not README_PATH.exists():
        errors.append("README.md is missing.")
    if not INDEX_PATH.exists():
        errors.append("docs/index.rst is missing.")
        return errors

    for entry in extract_toctree_entries(INDEX_PATH):
        if not (DOCS_DIR / f"{entry}.rst").exists():
            errors.append(f"docs/index.rst references missing document: {entry}.rst")

    if not API_PATH.exists():
        errors.append("docs/api.rst is missing.")
    elif "design_research_analysis" not in API_PATH.read_text(encoding="utf-8"):
        errors.append("docs/api.rst does not reference the package module.")
    return errors


def main() -> int:
    """Run the docs consistency check.

    Returns:
        Process exit code: `0` on success and `1` on failure.
    """
    errors = validate_docs_tree()
    if errors:
        for error in errors:
            print(error)
        return 1
    print("Documentation checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
