"""Devanagari captions render with the bundled Noto Sans Devanagari, never a system font.

The preset fonts (Anton, Inter) have no Devanagari, so Hindi text used to be
drawn by whatever font libass found on the machine — Nirmala UI on Windows,
which isn't licensed for redistribution with the project and varies by machine.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest
from autoclip.pipeline import captions, ffmpeg
from autoclip.pipeline.captions import (
    DEVANAGARI_FONT,
    DEVANAGARI_FONT_FILES,
    DEVANAGARI_SIZE_SCALE,
    FONT_DIR,
    build_ass,
    get_style,
    with_script_fonts,
)
from autoclip.pipeline.transcript import Word

HINDI = "मिशन"
SWITCH_IN = f"{{\\fn{DEVANAGARI_FONT}}}"


class TestScriptRuns:
    def test_latin_text_is_untouched(self) -> None:
        assert with_script_fonts("NOBODY TELLS YOU", "Anton") == "NOBODY TELLS YOU"

    def test_devanagari_word_is_wrapped_and_restored(self) -> None:
        assert with_script_fonts(HINDI, "Anton") == f"{SWITCH_IN}{HINDI}{{\\fnAnton}}"

    def test_trailing_ascii_punctuation_returns_to_the_preset_font(self) -> None:
        # The Devanagari font has no Latin; '?' must come from Anton.
        assert with_script_fonts("है?", "Anton") == f"{SWITCH_IN}है{{\\fnAnton}}?"

    def test_danda_stays_in_the_devanagari_font(self) -> None:
        # Anton has no danda.
        assert with_script_fonts("है।", "Anton") == f"{SWITCH_IN}है।{{\\fnAnton}}"

    def test_mixed_word_switches_per_run(self) -> None:
        assert with_script_fonts("GTAवाला", "Inter") == f"GTA{SWITCH_IN}वाला{{\\fnInter}}"

    def test_joiner_inside_a_conjunct_does_not_split_the_run(self) -> None:
        word = "क्‍ष"

        assert with_script_fonts(word, "Anton") == f"{SWITCH_IN}{word}{{\\fnAnton}}"

    def test_size_is_scaled_in_and_restored_out(self) -> None:
        scaled = round(100 * DEVANAGARI_SIZE_SCALE)

        assert with_script_fonts("है?", "Anton", 100) == (
            f"{{\\fn{DEVANAGARI_FONT}\\fs{scaled}}}है{{\\fnAnton\\fs100}}?"
        )


class TestPresets:
    words = [Word(HINDI, 0.0, 0.4), Word("OK", 0.5, 0.9), Word("है?", 1.0, 1.4)]

    @pytest.mark.parametrize("key", list(captions.PRESETS))
    def test_every_preset_switches_font_and_size_for_hindi(self, key: str) -> None:
        style = get_style(key)
        subs = build_ass(self.words, style, width=1080, height=1920)
        text = "\n".join(event.text for event in subs.events)
        base = round(1920 * style.size_ratio)
        enter = f"{{\\fn{DEVANAGARI_FONT}\\fs{round(base * DEVANAGARI_SIZE_SCALE)}}}"

        assert enter in text
        assert f"{{\\fn{style.font}\\fs{base}}}" in text
        # Latin never starts inside a Devanagari run.
        assert f"{enter}OK" not in text

    def test_english_output_is_unchanged(self) -> None:
        subs = build_ass([Word("hello", 0, 0.4)], get_style("bold_pop"), width=1080, height=1920)

        assert all("\\fn" not in event.text for event in subs.events)


class TestBundledFonts:
    def test_font_files_and_licence_are_bundled(self) -> None:
        for name in DEVANAGARI_FONT_FILES:
            assert (FONT_DIR / name).stat().st_size > 100_000
        assert "SIL Open Font License" in (FONT_DIR / "OFL-NotoSansDevanagari.txt").read_text(
            encoding="utf-8"
        )

    def test_family_name_matches_the_override(self) -> None:
        ttlib = pytest.importorskip("fontTools.ttLib")

        for name in DEVANAGARI_FONT_FILES:
            font = ttlib.TTFont(FONT_DIR / name)
            assert font["name"].getDebugName(1) == DEVANAGARI_FONT


@pytest.mark.slow
class TestRenderUsesOnlyBundledFonts:
    """Render real Hindi + English captions and read libass's font choices."""

    @pytest.mark.parametrize("key", ["bold_pop", "karaoke_fill", "clean_lower"])
    def test_no_system_font_fallback(self, tmp_path: Path, key: str) -> None:
        words = [
            Word("अभी", 0.0, 0.3),
            Word("मिशन", 0.3, 0.6),
            Word("पर", 0.6, 0.8),
            Word("ध्यान", 0.8, 1.1),
            Word("OK?", 1.2, 1.5),
            Word("है।", 1.5, 1.9),
        ]
        ass_path = captions.write_ass(
            tmp_path / "in.ass", words, get_style(key), width=1080, height=1920
        )
        workspace, subtitle_name, fonts_name = ffmpeg.relative_filter_workspace(ass_path, FONT_DIR)

        result = subprocess.run(
            [
                ffmpeg.ffmpeg_path(),
                "-hide_banner",
                "-loglevel",
                "verbose",
                "-f",
                "lavfi",
                "-i",
                "color=black:s=1080x1920:d=2:r=10",
                "-vf",
                f"ass=filename={subtitle_name}:fontsdir={fonts_name}",
                "-f",
                "null",
                "-",
            ],
            cwd=workspace,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
        )
        log = result.stderr

        chosen = set(re.findall(r"fontselect: \(([^,]+),[^)]*\) -> ([^,]+),", log))
        assert chosen, "libass selected no fonts; the test didn't exercise rendering"
        bundled = ("Anton", "Inter", "NotoSansDevanagari")
        for requested, resolved in chosen:
            assert resolved.startswith(bundled), f"{requested} fell back to {resolved}"
        assert "not found" not in log, "a glyph was missing from the chosen font"
