"""Overlay detectors: burned-in subtitles and facecams, on synthetic input."""

from __future__ import annotations

import numpy as np
import pytest
from autoclip.pipeline.reframe import overlays
from autoclip.pipeline.reframe.croppath import Rect
from autoclip.pipeline.reframe.faces import FaceObservation
from autoclip.pipeline.reframe.tracker import FaceTrack

W, H = 1920, 1080


def frame_with_text(top: int | None, *, seed: int = 0) -> np.ndarray:
    """Mid-grey noisy gameplay, with white outlined 'text' from ``top`` down."""
    rng = np.random.default_rng(seed)
    frame = rng.integers(90, 150, size=(H, W, 3), dtype=np.uint8)
    if top is not None:
        # Glyph-like strokes: white bars with black outlines, 40 px tall.
        for x in range(700, 1220, 24):
            frame[top - 3 : top + 43, x - 3 : x + 13] = 0
            frame[top : top + 40, x : x + 10] = 255
    return frame


class TestSubtitles:
    def test_finds_a_text_block(self) -> None:
        top = overlays.text_block_top(frame_with_text(960))
        assert top is not None and abs(top - 957) <= 4

    def test_plain_footage_has_no_text(self) -> None:
        assert overlays.text_block_top(frame_with_text(None)) is None

    def test_text_high_in_the_frame_is_ignored(self) -> None:
        # Titles and HUD text up top are not subtitles.
        assert overlays.text_block_top(frame_with_text(300)) is None


def test_band_sits_above_a_recurring_line() -> None:
    frames = [frame_with_text(960 if i % 2 else None, seed=i) for i in range(24)]
    band = overlays.detect_subtitle_band(frames)
    assert band is not None
    assert 920 <= band < 957


def test_no_band_without_subtitles() -> None:
    frames = [frame_with_text(None, seed=i) for i in range(12)]
    assert overlays.detect_subtitle_band(frames) is None


def test_one_off_text_is_not_a_band() -> None:
    # A sign in the world seen once in 24 frames.
    frames = [frame_with_text(960 if i == 3 else None, seed=i) for i in range(24)]
    assert overlays.detect_subtitle_band(frames) is None


class TestBandFromTops:
    def test_anchors_on_the_most_common_height(self) -> None:
        # Subtitles at 960; one-off lettering at 800 and 1000.
        tops = [960, 962, 958, 961, 800, 1000]
        band = overlays.subtitle_band_from_tops(tops, frame_count=24, height=H)
        assert band is not None and 940 <= band < 958

    def test_second_line_extends_the_band(self) -> None:
        tops = [960, 960, 961, 959, 900]
        band = overlays.subtitle_band_from_tops(tops, frame_count=24, height=H)
        assert band is not None and band < 900

    def test_too_rare_is_none(self) -> None:
        assert overlays.subtitle_band_from_tops([960], frame_count=24, height=H) is None

    def test_empty_is_none(self) -> None:
        assert overlays.subtitle_band_from_tops([], frame_count=24, height=H) is None


# --------------------------------------------------------------------------
# Facecam
# --------------------------------------------------------------------------


def seat(
    track_id: int,
    cx: float,
    cy: float,
    times: list[float],
    *,
    size: float = 110,
    drift: float = 0.0,
) -> FaceTrack:
    return FaceTrack(
        id=track_id,
        observations=[
            FaceObservation(
                t=t,
                cx=cx + drift * ((i % 5) - 2),
                cy=cy,
                width=size,
                height=size * 1.3,
                eye_y=cy - size * 0.2,
                mar=0.05,
            )
            for i, t in enumerate(times)
        ],
    )


def every(start: float, end: float, step: float = 0.2) -> list[float]:
    return [round(start + i * step, 3) for i in range(int((end - start) / step))]


SAMPLED = 150  # 30 s at 5 fps


def detect(tracks: list[FaceTrack]) -> Rect | None:
    return overlays.detect_facecam(tracks, source_w=W, source_h=H, sampled_frames=SAMPLED)


class TestFacecam:
    def test_two_steady_faces_in_a_corner(self) -> None:
        rect = detect([seat(1, 1550, 180, every(0, 30)), seat(2, 1760, 180, every(0, 30))])
        assert rect is not None
        assert rect.x < 1550 - 55 and rect.right > 1760 + 55
        assert rect.y < 180 - 70 and rect.bottom > 180 + 70

    def test_one_face_is_not_a_facecam(self) -> None:
        assert detect([seat(1, 1550, 180, every(0, 30))]) is None

    def test_fragmented_tracks_of_one_seat_merge(self) -> None:
        # The facecam drops out for B-roll, splitting each person into pieces.
        tracks = [
            seat(1, 1550, 180, every(0, 6)),
            seat(2, 1552, 181, every(10, 16)),
            seat(3, 1760, 180, every(0, 6)),
            seat(4, 1758, 179, every(10, 16)),
        ]
        assert detect(tracks) is not None

    def test_moving_faces_are_not_a_facecam(self) -> None:
        tracks = [
            seat(1, 900, 500, every(0, 30), drift=60),
            seat(2, 1100, 500, every(0, 30), drift=60),
        ]
        assert detect(tracks) is None

    def test_full_frame_people_are_too_big(self) -> None:
        tracks = [
            seat(1, 700, 450, every(0, 30), size=330),
            seat(2, 1200, 450, every(0, 30), size=330),
        ]
        assert detect(tracks) is None

    def test_brief_close_up_is_not_a_fixture(self) -> None:
        # Two game characters on screen for 3 s of a 30 s clip.
        tracks = [seat(1, 800, 400, every(10, 13)), seat(2, 1000, 400, every(10, 13))]
        assert detect(tracks) is None

    def test_people_never_on_screen_together_are_not_a_facecam(self) -> None:
        tracks = [seat(1, 1550, 180, every(0, 15)), seat(2, 1760, 180, every(15, 30))]
        assert detect(tracks) is None

    def test_supporting_seat_needs_to_be_beside_the_anchor(self) -> None:
        tracks = [seat(1, 300, 180, every(0, 30)), seat(2, 1700, 180, every(0, 30))]
        assert detect(tracks) is None

    def test_nothing_sampled(self) -> None:
        assert (
            overlays.detect_facecam(
                [seat(1, 1550, 180, every(0, 30))], source_w=W, source_h=H, sampled_frames=0
            )
            is None
        )


class TestGeometry:
    def test_faces_in(self) -> None:
        region = Rect(100, 100, 200, 200)
        inside = FaceObservation(t=0, cx=150, cy=150, width=50, height=60, eye_y=140, mar=0)
        outside = FaceObservation(t=0, cx=500, cy=150, width=50, height=60, eye_y=140, mar=0)
        assert overlays.faces_in(region, [inside])
        assert not overlays.faces_in(region, [inside, outside])
        assert not overlays.faces_in(region, [])

    @pytest.mark.parametrize("aspect", [9 / 16, 1.0, 1.6, 16 / 9])
    def test_fit_aspect_grows_and_keeps_the_region(self, aspect: float) -> None:
        region = Rect(1450, 61, 421, 259)
        rect = overlays.fit_aspect(region, aspect, bounds_w=W, bounds_h=H)
        assert rect.x <= region.x and rect.right >= region.right
        assert rect.y <= region.y and rect.bottom >= region.bottom
        assert rect.right <= W and rect.bottom <= H
        assert rect.width % 2 == 0 and rect.height % 2 == 0
        assert abs(rect.width / rect.height - aspect) < 0.02

    def test_fit_aspect_is_capped_by_the_bounds(self) -> None:
        rect = overlays.fit_aspect(Rect(0, 0, 1900, 100), 0.5, bounds_w=W, bounds_h=H)
        assert rect.height <= H and rect.width <= W
