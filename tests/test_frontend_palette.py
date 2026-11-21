"""Every colour utility the UI uses must exist in the theme palette.

Tailwind v4 generates a colour class only for tokens defined under ``@theme``
(``--color-<name>``). A class naming an undefined token compiles to nothing and
the element silently falls back to an inherited or transparent colour. That is
how the review page's 7/10 score bar rendered empty: it used ``bg-ink-500``
before ``--color-ink-500`` existed.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

FRONTEND_SRC = Path(__file__).resolve().parents[1] / "frontend" / "src"

#: The project's own colour families, as opposed to Tailwind's built-ins.
FAMILIES = ("ink", "sodium", "signal")

_UTILITY = re.compile(
    r"(?<![\w-])(?:[a-z]+:)*"  # optional variants such as hover: or group-hover:
    r"(?:bg|text|border(?:-[lrtbxy])?|ring|outline|fill|stroke|from|via|to|divide|"
    r"decoration|accent|caret|shadow|placeholder)"
    rf"-((?:{'|'.join(FAMILIES)})-[a-z0-9]+)"
    r"(?:/\d+)?(?![\w-])"
)


def defined_tokens() -> set[str]:
    css = (FRONTEND_SRC / "index.css").read_text(encoding="utf-8")
    return set(re.findall(r"--color-([a-z0-9-]+)\s*:", css))


def used_tokens() -> dict[str, set[str]]:
    used: dict[str, set[str]] = {}
    for path in FRONTEND_SRC.rglob("*.tsx"):
        for match in _UTILITY.finditer(path.read_text(encoding="utf-8")):
            used.setdefault(match.group(1), set()).add(path.name)
    return used


def test_sources_are_present() -> None:
    if not FRONTEND_SRC.exists():
        pytest.skip("frontend sources not available (installed from a wheel)")
    assert used_tokens(), "found no colour utilities; the pattern no longer matches the code"


def test_every_colour_utility_has_a_palette_token() -> None:
    if not FRONTEND_SRC.exists():
        pytest.skip("frontend sources not available (installed from a wheel)")

    defined = defined_tokens()
    missing = {
        token: sorted(files) for token, files in used_tokens().items() if token not in defined
    }

    assert not missing, f"colour classes with no --color-* token in index.css: {missing}"


@pytest.mark.parametrize(
    ("snippet", "token"),
    [
        ("'h-full bg-ink-500'", "ink-500"),
        ('className="text-ink-500 hover:text-sodium-400"', "sodium-400"),
        ("bg-ink-850/70", "ink-850"),
        ("group-hover:text-signal-good", "signal-good"),
    ],
)
def test_pattern_recognises_class_forms(snippet: str, token: str) -> None:
    assert token in {m.group(1) for m in _UTILITY.finditer(snippet)}
