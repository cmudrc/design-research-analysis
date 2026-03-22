"""Smoke-test the built wheel in an isolated virtual environment."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Install the built wheel in a temporary venv and smoke-test import + CLI.",
    )
    parser.add_argument(
        "--wheel",
        help="Path to the wheel file. Defaults to the single wheel in dist/.",
    )
    parser.add_argument(
        "--package-name",
        required=True,
        help="Installed distribution name used for metadata checks.",
    )
    parser.add_argument(
        "--import-name",
        required=True,
        help="Python import name to validate inside the temporary environment.",
    )
    parser.add_argument(
        "--cli",
        required=True,
        help="Console script name to invoke with --help.",
    )
    parser.add_argument(
        "--required-attr",
        help="Public attribute that must exist after import.",
    )
    return parser.parse_args()


def resolve_wheel(raw_wheel: str | None) -> Path:
    """Resolve the wheel path, defaulting to the single file in dist/."""
    if raw_wheel:
        wheel_path = Path(raw_wheel).expanduser().resolve()
        if not wheel_path.is_file():
            raise FileNotFoundError(f"Wheel not found: {wheel_path}")
        return wheel_path

    wheels = sorted(Path("dist").glob("*.whl"))
    if not wheels:
        raise FileNotFoundError("No wheel files found in dist/. Run the build step first.")
    if len(wheels) > 1:
        names = ", ".join(wheel.name for wheel in wheels)
        raise ValueError(f"Expected one wheel in dist/, found multiple: {names}")
    return wheels[0].resolve()


def venv_python(venv_root: Path) -> Path:
    """Return the Python executable inside the temporary environment."""
    if sys.platform.startswith("win"):
        return venv_root / "Scripts" / "python.exe"
    return venv_root / "bin" / "python"


def venv_cli(venv_root: Path, cli_name: str) -> Path:
    """Return the CLI entrypoint path inside the temporary environment."""
    if sys.platform.startswith("win"):
        scripts_dir = venv_root / "Scripts"
        for suffix in (".exe", ".cmd", ".bat", ""):
            candidate = scripts_dir / f"{cli_name}{suffix}"
            if candidate.exists():
                return candidate
        return scripts_dir / f"{cli_name}.exe"
    return venv_root / "bin" / cli_name


def run(command: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    """Run a subprocess and return its completed result."""
    return subprocess.run(
        command,
        check=True,
        capture_output=True,
        cwd=str(cwd) if cwd is not None else None,
        text=True,
    )


def base_python() -> Path:
    """Return the base interpreter used to create the temporary venv."""
    candidate = Path(getattr(sys, "_base_executable", sys.executable)).resolve()
    if candidate.is_file():
        return candidate
    return Path(sys.executable).resolve()


def main() -> int:
    """Create a temporary venv, install the wheel, and validate it."""
    args = parse_args()
    wheel_path = resolve_wheel(args.wheel)

    with tempfile.TemporaryDirectory(prefix="design-research-analysis-release-") as tmpdir:
        venv_root = Path(tmpdir) / "venv"
        run([str(base_python()), "-m", "venv", "--clear", str(venv_root)])

        python_path = venv_python(venv_root)
        cli_path = venv_cli(venv_root, args.cli)

        run([str(python_path), "-m", "pip", "install", "--upgrade", "pip"])
        run(
            [
                str(python_path),
                "-m",
                "pip",
                "install",
                str(wheel_path),
            ]
        )

        import_check_lines = [
            "import importlib.metadata as metadata",
            "import json",
            "from pathlib import Path",
            "",
            f"package = __import__({args.import_name!r})",
            "module_path = Path(package.__file__).resolve()",
            f"expected_root = Path({str(venv_root)!r}).resolve()",
            'module_version = getattr(package, "__version__", None)',
            f"distribution_version = metadata.version({args.package_name!r})",
            "payload = {",
            '    "distribution_version": distribution_version,',
            '    "module_path": str(module_path),',
            '    "module_version": module_version,',
            "}",
            "print(json.dumps(payload, sort_keys=True))",
            "",
            "if expected_root not in module_path.parents:",
            "    raise SystemExit(",
            '        "Imported module path "',
            '        f"{module_path} is not installed in the temp venv "',
            '        f"{expected_root}."',
            "    )",
            "if not isinstance(module_version, str) or not module_version:",
            '    raise SystemExit("Package is missing a non-empty __version__ export.")',
            "if module_version != distribution_version:",
            "    raise SystemExit(",
            '        "Module __version__ "',
            '        f"{module_version!r} does not match installed metadata "',
            '        f"{distribution_version!r}."',
            "    )",
        ]
        if args.required_attr:
            import_check_lines.extend(
                [
                    f"if not hasattr(package, {args.required_attr!r}):",
                    "    raise SystemExit(",
                    f'        "Missing required public attribute: {args.required_attr}"',
                    "    )",
                ]
            )
        import_check = "\n".join(import_check_lines)
        import_result = run([str(python_path), "-c", import_check])
        help_result = run([str(cli_path), "--help"])

        print(f"Wheel: {wheel_path.name}")
        print(f"Import smoke: {import_result.stdout.strip()}")
        first_help_line = help_result.stdout.splitlines()[0] if help_result.stdout else ""
        print(f"CLI smoke: {first_help_line}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
