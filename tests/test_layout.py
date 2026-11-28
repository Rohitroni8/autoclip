"""Layout rules: stacked facecam, crops above source subtitles, full-width opening."""

from __future__ import annotations

from autoclip.pipeline.reframe import layout
from autoclip.pipeline.reframe.croppath import (
    CropKeyframe,
    CropPath,
    CropSegment,
    Rect,
    Strategy,
)
from autoclip.pipeline.reframe.faces import FaceObservation
from autoclip.pipeline.reframe.tracker import FaceTrack

W, H = 1920, 1080
ASPECT = 9 / 16
#: Two webcam faces in the top-right corner.
FACECAM = Rect(1450, 60, 420, 260)


def face(t: float, cx: float, cy: float, size: float = 120) -> FaceObservation:
    return FaceObservation(
        t=t, cx=cx, cy=cy, width=size, height=size * 1.3, eye_y=cy - size * 0.2, mar=0.05
    )


def track(track_id: int, start: float, end: float, cx: float, cy: float) -> FaceTrack:
    count = int((end - start) * 5)
    return FaceTrack(
        id=track_id, observations=[face(start + i * 0.2, cx, cy) for i in range(count)]
    )


def segment(
    start: float, end: float, strategy: Strategy = Strategy.GENERAL, *, x: float = 656
) -> CropSegment:
    return CropSegment(
        start_s=start,
        end_s=end,
        width=608,
        height=1080,
        keyframes=[CropKeyframe(start, x, 0)],
        strategy=strategy,
    )


def apply(segments, tracks=(), facecam=None, usable_h=H):
    return layout.apply_layout(
        segments,
        tracks=list(tracks),
        facecam=facecam,
        usable_h=usable_h,
        source_w=W,
        source_h=H,
        aspect=ASPECT,
    )


def facecam_tracks(start: float, end: float) -> list[FaceTrack]:
    return [track(1, start, end, 1560, 180), track(2, start, end, 1760, 180)]


def assert_tiles(segments: list[CropSegment], duration: float) -> None:
    assert segments[0].start_s == 0.0
    assert segments[-1].end_s == duration
    for a, b in zip(segments, segments[1:], strict=False):
        assert a.end_s == b.start_s


class TestStacked:
    def test_facecam_segment_becomes_stacked(self) -> None:
        result = apply([segment(5, 20, Strategy.WIDE)], facecam_tracks(5, 20), FACECAM)
        assert [s.strategy for s in result] == [Strategy.STACKED]
        stacked = result[0]
        assert stacked.stack_split == layout.STACK_SPLIT
        top = stacked.stack_top
        assert top is not None
        # The band holds every facecam face, at the band's shape.
        assert top.x <= FACECAM.x and top.right >= FACECAM.right
        assert top.y <= FACECAM.y and top.bottom >= FACECAM.bottom
        assert abs(top.width / top.height - ASPECT / layout.STACK_SPLIT) < 0.02

    def test_gameplay_window_fills_the_bottom_band(self) -> None:
        stacked = apply([segment(5, 20)], facecam_tracks(5, 20), FACECAM)[0]
        window = stacked.window
        assert window.height == H
        assert abs(window.width / window.height - ASPECT / (1 - layout.STACK_SPLIT)) < 0.01

    def test_gameplay_window_moves_off_the_facecam(self) -> None:
        # Right of centre: a centred window would show the facecam a second time.
        facecam = Rect(1100, 40, 250, 220)
        tracks = [track(1, 0, 20, 1160, 150), track(2, 0, 20, 1290, 150)]
        window = apply([segment(5, 20)], tracks, facecam)[0].window
        assert window.right <= facecam.x

    def test_gameplay_window_stays_centred_without_room_beside_the_facecam(self) -> None:
        facecam = Rect(800, 40, 320, 220)
        tracks = [track(1, 0, 20, 900, 150), track(2, 0, 20, 1020, 150)]
        window = apply([segment(5, 20)], tracks, facecam)[0].window
        assert abs(window.x + window.width / 2 - W / 2) <= 1

    def test_segment_without_facecam_faces_is_left_alone(self) -> None:
        # A person in the middle of the frame, nowhere near the facecam.
        tracks = [track(1, 5, 20, 960, 500)]
        result = apply([segment(5, 20, Strategy.TRACK)], tracks, FACECAM)
        assert result[0].strategy is Strategy.TRACK

    def test_no_facecam_means_no_stacking(self) -> None:
        result = apply([segment(5, 20, Strategy.WIDE)], facecam_tracks(5, 20), None)
        assert result[0].strategy is Strategy.WIDE

    def test_short_faceless_gap_stays_stacked_and_merges(self) -> None:
        tracks = [*facecam_tracks(5, 10), *facecam_tracks(11.5, 20)]
        segments = [segment(5, 10), segment(10, 11.5), segment(11.5, 20)]
        result = apply(segments, tracks, FACECAM)
        assert [s.strategy for s in result] == [Strategy.STACKED]
        assert (result[0].start_s, result[0].end_s) == (5, 20)

    def test_long_faceless_gap_is_not_stacked(self) -> None:
        tracks = [*facecam_tracks(5, 10), *facecam_tracks(15, 20)]
        segments = [segment(5, 10), segment(10, 15), segment(15, 20)]
        result = apply(segments, tracks, FACECAM)
        assert [s.strategy for s in result] == [
            Strategy.STACKED,
            Strategy.GENERAL,
            Strategy.STACKED,
        ]

    def test_landscape_output_is_never_stacked(self) -> None:
        result = layout.apply_layout(
            [segment(5, 20)],
            tracks=facecam_tracks(5, 20),
            facecam=FACECAM,
            usable_h=H,
            source_w=W,
            source_h=H,
            aspect=16 / 9,
        )
        assert result[0].strategy is Strategy.GENERAL


class TestSubtitles:
    def test_tall_crop_shrinks_above_the_subtitles(self) -> None:
        result = apply([segment(5, 20, Strategy.TRACK, x=600)], usable_h=900)
        crop = result[0]
        assert crop.height <= 900
        assert crop.keyframes[0].y + crop.height <= 900
        assert abs(crop.width / crop.height - ASPECT) < 0.01
        # Same centre as before.
        assert abs((crop.keyframes[0].x + crop.width / 2) - (600 + 608 / 2)) <= 1

    def test_fit_window_stops_at_the_subtitles(self) -> None:
        fit = CropSegment(
            start_s=5,
            end_s=20,
            width=W,
            height=H,
            keyframes=[CropKeyframe(5, 0, 0)],
            strategy=Strategy.WIDE,
            fit=True,
        )
        result = apply([fit], usable_h=900)
        assert result[0].window == Rect(0, 0, W, 900)

    def test_stacked_windows_stay_above_the_subtitles(self) -> None:
        stacked = apply([segment(5, 20)], facecam_tracks(5, 20), FACECAM, usable_h=900)[0]
        assert stacked.window.bottom <= 900
        assert stacked.stack_top is not None and stacked.stack_top.bottom <= 900

    def test_no_subtitles_changes_nothing(self) -> None:
        original = segment(5, 20, Strategy.TRACK, x=600)
        assert apply([original])[0] == original


class TestOpening:
    def test_faceless_opening_is_fitted_full_width(self) -> None:
        result = apply([segment(0, 10)], usable_h=900)
        assert_tiles(result, 10)
        opening, rest = result
        assert opening.fit and opening.end_s == layout.OPENING_S
        assert opening.window == Rect(0, 0, W, 900)
        assert not rest.fit and rest.keyframes[0].t == layout.OPENING_S

    def test_wide_opening_is_fitted(self) -> None:
        result = apply([segment(0, 10, Strategy.WIDE)])
        assert result[0].fit and result[0].strategy is Strategy.WIDE

    def test_tracked_opening_keeps_its_framing(self) -> None:
        original = segment(0, 10, Strategy.TRACK)
        assert apply([original]) == [original]

    def test_stacked_opening_shows_the_full_gameplay_width(self) -> None:
        result = apply([segment(0, 10)], facecam_tracks(0, 10), FACECAM)
        assert_tiles(result, 10)
        opening, rest = result
        assert opening.strategy is Strategy.STACKED and not opening.fit
        assert opening.window == Rect(0, 0, W, H)
        assert opening.stack_top == rest.stack_top
        assert rest.window.width < W

    def test_short_remainder_is_absorbed(self) -> None:
        result = apply([segment(0, layout.OPENING_S + 0.5), segment(layout.OPENING_S + 0.5, 10)])
        assert result[0].fit and result[0].end_s == layout.OPENING_S + 0.5
        assert_tiles(result, 10)

    def test_every_segment_inside_the_opening_is_widened(self) -> None:
        result = apply([segment(0, 1.5), segment(1.5, 10, Strategy.WIDE)])
        assert [s.fit for s in result] == [True, True, False]
        assert_tiles(result, 10)


def test_stacked_segment_round_trips() -> None:
    stacked = apply([segment(5, 20)], facecam_tracks(5, 20), FACECAM)
    path = CropPath(source_width=W, source_height=H, segments=stacked)
    assert CropPath.from_dict(path.to_dict()) == path
