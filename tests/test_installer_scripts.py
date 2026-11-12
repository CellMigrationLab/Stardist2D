"""The installer scripts must name this app: no template placeholder may survive initialisation."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PLACEHOLDER = re.compile(r"PYTHON_PROJ_NAME|UNDERSCORED_PROJECT_NAME|PROJECT_NAME|VERSION_NUMBER|PYTHON_VERSION")
INSTALLED_FILES = [
    "app/bash_bat_scripts/post_install.sh",
    "app/bash_bat_scripts/post_install.bat",
    "app/bash_bat_scripts/pre_uninstall.sh",
    "app/bash_bat_scripts/pre_uninstall.bat",
    "app/bash_bat_scripts/uninstall.sh",
    "app/python_scripts/launch_jupyter.py",
    "app/menuinst/notebook_launcher.json",
    "setup.py",
]


@pytest.mark.parametrize("relative", INSTALLED_FILES)
def test_no_template_placeholder_left(relative: str) -> None:
    """Fail on a placeholder: 1.0.0's post_install ran ``import PYTHON_PROJ_NAME`` and every installation failed."""
    path = ROOT / relative
    if not path.is_file():
        pytest.fail(f"{relative} is missing; update INSTALLED_FILES if the template moved it")
    found = sorted({match.group(0) for match in PLACEHOLDER.finditer(path.read_text(encoding="utf-8"))})
    assert not found, f"{relative} still contains {found}"


def test_post_install_checks_this_package() -> None:
    """The import check and tool registration of post_install name the ``stardist2d`` package."""
    for script in ("post_install.sh", "post_install.bat"):
        text = (ROOT / "app/bash_bat_scripts" / script).read_text(encoding="utf-8")
        assert "import stardist2d;" in text, script
        assert "stardist2d_lc_tools" in text, script


def test_windows_script_keeps_crlf_line_endings() -> None:
    """post_install.bat has CR LF line endings, as in the template: cmd.exe can miss GOTO/CALL labels otherwise."""
    data = (ROOT / "app/bash_bat_scripts/post_install.bat").read_bytes()
    assert data.count(b"\n") == data.count(b"\r\n"), "post_install.bat has lines ending in LF only"
