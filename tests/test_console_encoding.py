from __future__ import annotations

from scripts import console_encoding


class FakeStream:
    def __init__(self, tty: bool = True) -> None:
        self.tty = tty
        self.calls: list[dict[str, str]] = []

    def isatty(self) -> bool:
        return self.tty

    def reconfigure(self, **kwargs: str) -> None:
        self.calls.append(kwargs)


def test_configure_windows_console_encoding_uses_active_code_page(monkeypatch) -> None:
    stdout = FakeStream()
    stderr = FakeStream()
    monkeypatch.setattr(console_encoding.os, "name", "nt", raising=False)
    monkeypatch.setattr(console_encoding, "_get_console_output_code_page", lambda: 949)

    encoding = console_encoding.configure_windows_console_encoding(stdout, stderr)

    assert encoding == "cp949"
    assert stdout.calls == [{"encoding": "cp949", "errors": "replace"}]
    assert stderr.calls == [{"encoding": "cp949", "errors": "replace"}]


def test_configure_windows_console_encoding_keeps_utf8_code_page(monkeypatch) -> None:
    stdout = FakeStream()
    stderr = FakeStream()
    monkeypatch.setattr(console_encoding.os, "name", "nt", raising=False)
    monkeypatch.setattr(console_encoding, "_get_console_output_code_page", lambda: 65001)

    encoding = console_encoding.configure_windows_console_encoding(stdout, stderr)

    assert encoding == "utf-8"
    assert stdout.calls == [{"encoding": "utf-8", "errors": "replace"}]
    assert stderr.calls == [{"encoding": "utf-8", "errors": "replace"}]


def test_configure_windows_console_encoding_skips_non_tty(monkeypatch) -> None:
    stdout = FakeStream(tty=False)
    stderr = FakeStream(tty=False)
    monkeypatch.setattr(console_encoding.os, "name", "nt", raising=False)
    monkeypatch.setattr(console_encoding, "_get_console_output_code_page", lambda: 949)

    encoding = console_encoding.configure_windows_console_encoding(stdout, stderr)

    assert encoding == "cp949"
    assert stdout.calls == []
    assert stderr.calls == []


def test_configure_windows_console_encoding_skips_non_windows(monkeypatch) -> None:
    stdout = FakeStream()
    stderr = FakeStream()
    monkeypatch.setattr(console_encoding.os, "name", "posix", raising=False)

    encoding = console_encoding.configure_windows_console_encoding(stdout, stderr)

    assert encoding is None
    assert stdout.calls == []
    assert stderr.calls == []
