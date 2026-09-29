"""Portability / dependency hygiene checks for the repository itself."""

import ast
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
LOCAL_PACKAGES = {"config", "features", "services", "training", "models", "tests"}

# import name -> distribution name in requirements.txt
DIST_NAMES = {"sklearn": "scikit-learn", "dotenv": "python-dotenv"}


# ==========================================
# config/ vs Config/ (case-sensitive filesystems)
# ==========================================


def test_config_directory_is_lowercase_on_disk():
    names = {p.name for p in ROOT.iterdir()}
    assert "config" in names
    assert "Config" not in names


@pytest.mark.skipif(shutil.which("git") is None, reason="git not available")
def test_git_tracks_config_with_lowercase_path():
    """Imports use `config.settings`; a tracked `Config/` path breaks Linux/macOS clones."""
    result = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True)
    tracked = result.stdout.splitlines()
    assert "config/settings.py" in tracked
    assert not [p for p in tracked if p.startswith("Config/")]


# ==========================================
# requirements.txt
# ==========================================


def _requirement_names() -> set[str]:
    raw = (ROOT / "requirements.txt").read_bytes()
    assert not raw.startswith((b"\xff\xfe", b"\xfe\xff")), "requirements.txt must not be UTF-16"
    text = raw.decode("utf-8")
    names = set()
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            names.add(line.split("==")[0].strip().lower())
    return names


def _project_py_files():
    for top in ROOT.iterdir():
        if top.name in {".venv", "venv", "env", ".git"}:
            continue
        if top.is_file() and top.suffix == ".py":
            yield top
        elif top.is_dir():
            yield from top.rglob("*.py")


def _third_party_imports() -> set[str]:
    found = set()
    for path in _project_py_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                mods = [node.module]
            else:
                continue
            for mod in mods:
                top = mod.split(".")[0]
                if top not in sys.stdlib_module_names and top not in LOCAL_PACKAGES:
                    found.add(top)
    return found


def test_requirements_utf8_and_pinned():
    text = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    reqs = [l.split("#", 1)[0].strip() for l in text.splitlines() if l.split("#", 1)[0].strip()]
    assert reqs and all("==" in r for r in reqs)


def test_every_imported_package_is_declared():
    declared = _requirement_names()
    missing = {m for m in _third_party_imports() if DIST_NAMES.get(m, m).lower() not in declared}
    assert not missing, f"imported but not in requirements.txt: {sorted(missing)}"


# ==========================================
# app.py placeholder
# ==========================================


def test_app_py_is_an_explicit_placeholder():
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(str(ROOT / "app.py"), run_name="__main__")
    assert "reserved for the Flask backend" in str(exc.value.code)
