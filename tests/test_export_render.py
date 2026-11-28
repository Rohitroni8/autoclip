"""End-to-end render tests against real ffmpeg.

These are the tests that catch what unit tests structurally cannot: a
filtergraph that parses but produces the wrong thing, a font libass can't
resolve, a path escaping bug that only appears on Windows, an encoder flag the
local build rejects. Marked ``slow`` — they shell out and encode video.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from autoclip.config import ExportSettings
from autoclip.pipeline import captions, export, ffmpeg
from autoclip.pipeline.reframe import layout
from autoclip.pipeline.reframe.croppath import (
    CropKeyframe,
    CropPath,
    CropSegment,
    Rect,
    Strategy,
    centre_crop,
)
from autoclip.pipeline.transcript import Word

pytestmark = pytest.mark.slow

SOURCE_W, SOURCE_H = 1280, 720
SOURCE_DURATION = 10.0


@pytest.fixture(scope="module")
def source_video(tmp_path_factory) -> Path:
    """A synthetic 1280x720 test clip with a tone, generated once per session."""
    path = tmp_path_factory.mktemp("media") / "source.mp4"
    subprocess.run(
        [
            ffmpeg.ffmpeg_path(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=size={SOURCE_W}x{SOURCE_H}:rate=30:duration={SOURCE_DURATION}",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:duration={SOURCE_DURATION}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    return path


@pytest.fixture
def words() -> list[Word]:
    """Words spanning 2.0-7.0s of the source, matching the clip under test."""
    texts = [
        "this",
        "is",
        "a",
        "test",
        "of",
        "the",
        "caption",
        "rendering",
        "pipeline",
        "right",
        "now",
    ]
    step = 5.0 / len(texts)
    return [
        Word(text=text, start=2.0 + i * step, end=2.0 + i * step + step * 0.85)
        for i, text in enumerate(texts)
    ]


def make_request(source: Path, destination: Path, crop_path: CropPath, words, **kwargs):
    return export.ExportRequest(
        source=source,
        destination=destination,
        start_s=2.0,
        end_s=7.0,
        crop_path=crop_path,
        words=words,
        style=kwargs.pop("style", captions.get_style("bold_pop")),
        **kwargs,
    )


class TestSingleSegmentRender:
    def test_renders_a_vertical_clip(self, source_video, words, tmp_path) -> None:
        destination = tmp_path / "out.mp4"
        request = make_request(
            source_video, destination, centre_crop(SOURCE_W, SOURCE_H, 5.0), words
        )

        export.export_clip(request, work_dir=tmp_path / "work")

        assert destination.exists()
        info = ffmpeg.probe(destination)
        assert (info.width, info.height) == (1080, 1920)
        assert info.has_audio
        assert info.duration_s == pytest.approx(5.0, abs=0.35)

    @pytest.mark.parametrize("ratio", ["9:16", "1:1", "16:9"])
    def test_every_ratio_renders(self, source_video, words, tmp_path, ratio: str) -> None:
        destination = tmp_path / f"out_{ratio.replace(':', 'x')}.mp4"
        aspect = tuple(int(part) for part in ratio.split(":"))
        crop_path = centre_crop(SOURCE_W, SOURCE_H, 5.0, aspect_w=aspect[0], aspect_h=aspect[1])

        export.export_clip(
            make_request(source_video, destination, crop_path, words, ratio=ratio),
            work_dir=tmp_path / "work",
        )

        info = ffmpeg.probe(destination)
        assert (info.width, info.height) == export.ratio_dimensions(ratio)

    @pytest.mark.parametrize("style_key", list(captions.PRESETS))
    def test_every_caption_style_burns_in(
        self, source_video, words, tmp_path, style_key: str
    ) -> None:
        # A style whose font libass can't resolve still "succeeds" but renders
        # in a substituted face, so this checks the render completes for each.
        destination = tmp_path / f"out_{style_key}.mp4"
        request = make_request(
            source_video,
            destination,
            centre_crop(SOURCE_W, SOURCE_H, 5.0),
            words,
            style=captions.get_style(style_key),
        )

        export.export_clip(request, work_dir=tmp_path / "work")

        assert destination.stat().st_size > 1000

    def test_captions_visibly_change_the_output(self, source_video, words, tmp_path) -> None:
        """Burned captions must actually alter the pixels.

        Without this, a silently-failing `ass` filter would leave every other
        assertion passing while shipping clips with no captions on them.
        """
        with_captions = tmp_path / "with.mp4"
        without_captions = tmp_path / "without.mp4"

        export.export_clip(
            make_request(source_video, with_captions, centre_crop(SOURCE_W, SOURCE_H, 5.0), words),
            work_dir=tmp_path / "work",
        )
        export.export_clip(
            make_request(
                source_video,
                without_captions,
                centre_crop(SOURCE_W, SOURCE_H, 5.0),
                words,
                burn_captions=False,
            ),
            work_dir=tmp_path / "work",
        )

        assert _frame_signature(with_captions, 2.5) != _frame_signature(without_captions, 2.5)


class TestMultiSegmentRender:
    def test_concatenated_segments_render(self, source_video, words, tmp_path) -> None:
        crop_w, crop_h = 404, 720
        crop_path = CropPath(
            source_width=SOURCE_W,
            source_height=SOURCE_H,
            segments=[
                CropSegment(
                    start_s=0.0,
                    end_s=2.5,
                    width=crop_w,
                    height=crop_h,
                    keyframes=[CropKeyframe(0.0, 100.0, 0.0)],
                    strategy=Strategy.TRACK,
                ),
                CropSegment(
                    start_s=2.5,
                    end_s=5.0,
                    width=crop_w,
                    height=crop_h,
                    keyframes=[CropKeyframe(2.5, 700.0, 0.0)],
                    strategy=Strategy.TRACK,
                ),
            ],
        )
        destination = tmp_path / "multi.mp4"

        export.export_clip(
            make_request(source_video, destination, crop_path, words),
            work_dir=tmp_path / "work",
        )

        info = ffmpeg.probe(destination)
        assert (info.width, info.height) == (1080, 1920)
        assert info.duration_s == pytest.approx(5.0, abs=0.35)

    def test_segments_actually_show_different_regions(self, source_video, words, tmp_path) -> None:
        """A crop that doesn't move would make the two segments identical."""
        crop_w, crop_h = 404, 720
        crop_path = CropPath(
            source_width=SOURCE_W,
            source_height=SOURCE_H,
            segments=[
                CropSegment(
                    start_s=0.0,
                    end_s=2.5,
                    width=crop_w,
                    height=crop_h,
                    keyframes=[CropKeyframe(0.0, 0.0, 0.0)],
                ),
                CropSegment(
                    start_s=2.5,
                    end_s=5.0,
                    width=crop_w,
                    height=crop_h,
                    keyframes=[CropKeyframe(2.5, 876.0, 0.0)],
                ),
            ],
        )
        destination = tmp_path / "regions.mp4"

        export.export_clip(
            make_request(source_video, destination, crop_path, words, burn_captions=False),
            work_dir=tmp_path / "work",
        )

        assert _frame_signature(destination, 1.0) != _frame_signature(destination, 4.0)

    def test_animated_crop_expression_renders(self, source_video, words, tmp_path) -> None:
        crop_path = CropPath(
            source_width=SOURCE_W,
            source_height=SOURCE_H,
            segments=[
                CropSegment(
                    start_s=0.0,
                    end_s=5.0,
                    width=404,
                    height=720,
                    keyframes=[
                        CropKeyframe(0.0, 0.0, 0.0),
                        CropKeyframe(2.5, 400.0, 0.0),
                        CropKeyframe(5.0, 876.0, 0.0),
                    ],
                    strategy=Strategy.TRACK,
                )
            ],
        )
        destination = tmp_path / "animated.mp4"

        export.export_clip(
            make_request(source_video, destination, crop_path, words, burn_captions=False),
            work_dir=tmp_path / "work",
        )

        # A broken expression would evaluate to a constant, leaving these equal.
        assert _frame_signature(destination, 0.5) != _frame_signature(destination, 4.5)

    def test_fit_segment_renders_with_blurred_background(
        self, source_video, words, tmp_path
    ) -> None:
        crop_path = CropPath(
            source_width=SOURCE_W,
            source_height=SOURCE_H,
            segments=[
                CropSegment(
                    start_s=0.0,
                    end_s=5.0,
                    width=SOURCE_W,
                    height=SOURCE_H,
                    keyframes=[CropKeyframe(0.0, 0.0, 0.0)],
                    strategy=Strategy.WIDE,
                    fit=True,
                )
            ],
        )
        destination = tmp_path / "fit.mp4"

        export.export_clip(
            make_request(source_video, destination, crop_path, words),
            work_dir=tmp_path / "work",
        )

        info = ffmpeg.probe(destination)
        assert (info.width, info.height) == (1080, 1920)


class TestEncoding:
    def test_software_encoder_path(self, source_video, words, tmp_path) -> None:
        settings = ExportSettings(prefer_hardware_encoder=False)
        destination = tmp_path / "x264.mp4"

        export.export_clip(
            make_request(source_video, destination, centre_crop(SOURCE_W, SOURCE_H, 5.0), words),
            work_dir=tmp_path / "work",
            settings=settings,
        )

        assert ffmpeg.probe(destination).video_codec == "h264"

    def test_output_is_yuv420p(self, source_video, words, tmp_path) -> None:
        # Anything else fails to play on a surprising number of phones.
        destination = tmp_path / "pixfmt.mp4"
        export.export_clip(
            make_request(source_video, destination, centre_crop(SOURCE_W, SOURCE_H, 5.0), words),
            work_dir=tmp_path / "work",
        )

        result = subprocess.run(
            [
                ffmpeg.ffprobe_path(),
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=pix_fmt",
                "-of",
                "csv=p=0",
                str(destination),
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        assert result.stdout.strip() == "yuv420p"

    def test_srt_sidecar_is_written_when_requested(self, source_video, words, tmp_path) -> None:
        settings = ExportSettings(write_srt=True)
        destination = tmp_path / "sidecar.mp4"

        export.export_clip(
            make_request(source_video, destination, centre_crop(SOURCE_W, SOURCE_H, 5.0), words),
            work_dir=tmp_path / "work",
            settings=settings,
        )

        assert destination.with_suffix(".srt").exists()

    def test_progress_reaches_completion(self, source_video, words, tmp_path) -> None:
        seen: list[float] = []

        export.export_clip(
            make_request(
                source_video, tmp_path / "progress.mp4", centre_crop(SOURCE_W, SOURCE_H, 5.0), words
            ),
            work_dir=tmp_path / "work",
            on_progress=seen.append,
        )

        assert seen
        assert seen[-1] == 1.0
        assert all(0.0 <= value <= 1.0 for value in seen)


class TestPathsWithSpecialCharacters:
    def test_directory_with_spaces_and_punctuation(self, source_video, words, tmp_path) -> None:
        """The escaping test that actually matters on Windows.

        A drive letter plus a space plus an apostrophe is the combination that
        breaks naive filtergraph quoting.
        """
        awkward = tmp_path / "My Clips (2026)" / "it's here"
        awkward.mkdir(parents=True)
        destination = awkward / "out.mp4"

        export.export_clip(
            make_request(source_video, destination, centre_crop(SOURCE_W, SOURCE_H, 5.0), words),
            work_dir=awkward / "work",
        )

        assert destination.exists()
        assert ffmpeg.probe(destination).width == 1080


def _frame_signature(video: Path, timestamp: float) -> str:
    """Hash one frame's pixels, for comparing rendered output."""
    result = subprocess.run(
        [
            ffmpeg.ffmpeg_path(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            str(timestamp),
            "-i",
            str(video),
            "-frames:v",
            "1",
            "-f",
            "hash",
            "-hash",
            "md5",
            "-",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


# --------------------------------------------------------------------------
# Stacked layout and windowed fit
# --------------------------------------------------------------------------

#: A frame of solid regions: a red "facecam" top right, green "gameplay" in the
#: middle, a yellow "subtitle" strip at the bottom, blue everywhere else.
FACECAM_BOX = (960, 40, 280, 160)
GAMEPLAY_BOX = (380, 60, 520, 560)
SUBTITLE_TOP = 640


@pytest.fixture(scope="module")
def regions_video(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("media") / "regions.mp4"
    boxes = [
        (*FACECAM_BOX, "red"),
        (*GAMEPLAY_BOX, "lime"),
        (0, SUBTITLE_TOP, SOURCE_W, SOURCE_H - SUBTITLE_TOP, "yellow"),
    ]
    draw = ",".join(f"drawbox=x={x}:y={y}:w={w}:h={h}:color={c}:t=fill" for x, y, w, h, c in boxes)
    subprocess.run(
        [
            ffmpeg.ffmpeg_path(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c=blue:size={SOURCE_W}x{SOURCE_H}:rate=30:duration={SOURCE_DURATION},{draw}",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:duration={SOURCE_DURATION}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    return path


def _frame_rgb(video: Path, timestamp: float):
    """One decoded frame as an (H, W, 3) uint8 array."""
    import numpy as np

    info = ffmpeg.probe(video)
    result = subprocess.run(
        [
            ffmpeg.ffmpeg_path(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            str(timestamp),
            "-i",
            str(video),
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ],
        capture_output=True,
        check=True,
    )
    return np.frombuffer(result.stdout, np.uint8).reshape(info.height, info.width, 3)


def _dominant(pixels) -> str:
    """Name the colour most of ``pixels`` are closest to."""
    import numpy as np

    palette = {
        "red": (255, 0, 0),
        "lime": (0, 255, 0),
        "yellow": (255, 255, 0),
        "blue": (0, 0, 255),
    }
    flat = pixels.reshape(-1, 3).astype(int)
    names = list(palette)
    distances = np.stack([((flat - palette[n]) ** 2).sum(axis=1) for n in names])
    counts = np.bincount(distances.argmin(axis=0), minlength=len(names))
    return names[int(counts.argmax())]


def _red_pixels(pixels) -> int:
    return int(((pixels[..., 0] > 200) & (pixels[..., 1] < 60) & (pixels[..., 2] < 60)).sum())


class TestStackedRender:
    def _render(self, video: Path, segments, tmp_path, name: str) -> Path:
        crop_path = CropPath(source_width=SOURCE_W, source_height=SOURCE_H, segments=segments)
        destination = tmp_path / f"{name}.mp4"
        export.export_clip(
            make_request(video, destination, crop_path, [], burn_captions=False),
            work_dir=tmp_path / "work",
        )
        return destination

    def test_facecam_on_top_gameplay_below(self, regions_video, tmp_path) -> None:
        stacked = layout.stacked_segment(
            0.0,
            5.0,
            facecam=Rect(*FACECAM_BOX),
            usable_h=SUBTITLE_TOP,
            source_w=SOURCE_W,
            aspect=9 / 16,
        )
        destination = self._render(regions_video, [stacked], tmp_path, "stacked")

        info = ffmpeg.probe(destination)
        assert (info.width, info.height) == (1080, 1920)
        frame = _frame_rgb(destination, 1.0)
        top_h = 1920 * layout.STACK_SPLIT
        assert _dominant(frame[: int(top_h)]) == "red"
        assert _dominant(frame[int(top_h) :]) == "lime"
        # The source's subtitles never reach the output.
        assert _dominant(frame[-40:]) != "yellow"

    def test_full_width_opening_then_enlarged_gameplay(self, regions_video, tmp_path) -> None:
        kwargs = {
            "facecam": Rect(*FACECAM_BOX),
            "usable_h": SUBTITLE_TOP,
            "source_w": SOURCE_W,
            "aspect": 9 / 16,
        }
        opening = layout.stacked_segment(0.0, 2.0, full_width=True, **kwargs)
        rest = layout.stacked_segment(2.0, 5.0, **kwargs)
        destination = self._render(regions_video, [opening, rest], tmp_path, "stacked_opening")

        top = int(1920 * layout.STACK_SPLIT)
        # Full width: the whole frame, facecam corner included, shows below.
        # Enlarged: the gameplay alone, beside the facecam.
        assert _red_pixels(_frame_rgb(destination, 1.0)[top:]) > 5000
        assert _red_pixels(_frame_rgb(destination, 4.0)[top:]) == 0
        assert _dominant(_frame_rgb(destination, 4.0)[top:]) == "lime"

    def test_fit_window_leaves_out_the_subtitles(self, regions_video, tmp_path) -> None:
        fit = CropSegment(
            start_s=0.0,
            end_s=5.0,
            width=SOURCE_W,
            height=SUBTITLE_TOP,
            keyframes=[CropKeyframe(0.0, 0.0, 0.0)],
            strategy=Strategy.WIDE,
            fit=True,
        )
        destination = self._render(regions_video, [fit], tmp_path, "fit_window")
        frame = _frame_rgb(destination, 1.0)
        # The sharp middle band is the fitted window; no yellow in it.
        middle = frame[960 - 200 : 960 + 200]
        assert not ((middle[..., 0] > 200) & (middle[..., 1] > 200) & (middle[..., 2] < 80)).any()
