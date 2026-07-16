from __future__ import annotations

import ctypes
import os
import sys
from typing import TextIO


def configure_windows_console_encoding(
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> str | None:
    """Match Python text output to the active Windows console code page."""
    if os.name != "nt":
        return None

    code_page = _get_console_output_code_page()
    if code_page <= 0:
        return None

    encoding = "utf-8" if code_page == 65001 else f"cp{code_page}"
    for stream in (stdout or sys.stdout, stderr or sys.stderr):
        _reconfigure_tty_stream(stream, encoding)
    return encoding


def _get_console_output_code_page() -> int:
    try:
        return int(ctypes.windll.kernel32.GetConsoleOutputCP())
    except (AttributeError, OSError, ValueError):
        return 0


def _reconfigure_tty_stream(stream: TextIO, encoding: str) -> None:
    reconfigure = getattr(stream, "reconfigure", None)
    isatty = getattr(stream, "isatty", None)
    if not callable(reconfigure) or not callable(isatty) or not isatty():
        return
    try:
        reconfigure(encoding=encoding, errors="replace")
    except (OSError, ValueError):
        return
