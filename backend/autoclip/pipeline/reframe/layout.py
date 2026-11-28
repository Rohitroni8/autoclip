"""Layout rules applied on top of per-shot framing.

:func:`build_crop_path` frames each shot around faces. This module then adjusts
those segments for what is drawn over the picture (see :mod:`.overlays`):

1. **Stacked.** When the clip has a facecam, a segment showing facecam faces
   becomes STACKED: the facecam in a band on top, the gameplay enlarged below.
2. **Above the subtitles.** No crop reaches into the source's own burned-in
   subtitles.
3. **The opening.** The first :data:`OPENING_S` seconds show the full width of
   the frame (fitted, or as the stacked gameplay band), so titles drawn across
   the video ("HOUR 23") aren't cut. A tracked single subject keeps its framing.

Everything here is a pure function of segments, tracks and the detected
overlays, so it is tested without video.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from .croppath import CropKeyframe, CropSegment, Rect, Strategy
from .overlays import fit_aspect
from .tracker import FaceTrack

#: Length of the opening that shows the full frame width.
OPENING_S = 3.0
#: A segment left after splitting off the opening that is shorter than this is
#: absorbed into the opening rather than flashing on screen.
MIN_SEGMENT_S = 1.0
#: Fraction of the output height given to the facecam band.
STACK_SPLIT = 0.35
#: A segment with no faces this short, between two stacked segments, is a
#: facecam detection dropout rather than a different scene: keep it stacked.
STACK_GAP_FILL_S = 2.0
#: Observations inside the facecam a segment needs before it is stacked; one
#: stray detection isn't enough.
STACK_MIN_FACECAM_HITS = 2


def apply_layout(
    segments: Sequence[CropSegment],
    *,
    tracks: Sequence[FaceTrack],
    facecam: Rect | None,
    usable_h: int,
    source_w: int,
    source_h: int,
    aspect: float,
) -> list[CropSegment]:
    """Apply the stacked, subtitle and opening rules to tiled crop segments.

    ``usable_h`` is the source height above any burned-in subtitles (the full
    height when there are none). ``aspect`` is the output width over height.

    Preconditions:
        segments tile the clip in order, clip-relative.
    """
    usable_h = _even(min(usable_h, source_h))
    result = [replace(s) for s in segments]

    portrait = aspect < 1.0
    if facecam is not None and portrait:
        result = _stack(result, tracks, facecam, usable_h, source_w, aspect)

    result = [_above_subtitles(s, usable_h, source_w, aspect) for s in result]
    return _merge(_opening(result, usable_h, source_w, aspect))


# --------------------------------------------------------------------------
# Stacked
# --------------------------------------------------------------------------


def stacked_segment(
    start_s: float,
    end_s: float,
    *,
    facecam: Rect,
    usable_h: int,
    source_w: int,
    aspect: float,
    full_width: bool = False,
) -> CropSegment:
    """A STACKED segment: the facecam on top, the gameplay below.

    The gameplay window fills the bottom band at full usable height, beside the
    facecam where possible so it isn't shown twice. ``full_width`` shows the
    whole width of the gameplay instead (fitted into the band).
    """
    top = fit_aspect(facecam, aspect / STACK_SPLIT, bounds_w=source_w, bounds_h=usable_h)
    if full_width:
        window = Rect(0, 0, source_w, usable_h)
    else:
        width = min(source_w, _even(usable_h * aspect / (1 - STACK_SPLIT)))
        window = Rect(_gameplay_x(width, facecam, source_w), 0, width, usable_h)
    return CropSegment(
        start_s=start_s,
        end_s=end_s,
        width=int(window.width),
        height=int(window.height),
        keyframes=[CropKeyframe(t=start_s, x=window.x, y=window.y)],
        strategy=Strategy.STACKED,
        stack_top=top,
        stack_split=STACK_SPLIT,
    )


def _gameplay_x(width: float, facecam: Rect, source_w: int) -> float:
    """Left edge of the gameplay window: centred, unless that shows the facecam.

    Moves to whichever side of the facecam keeps the window nearest the centre;
    if neither side has room, stays centred.
    """
    centred = (source_w - width) / 2
    if centred + width <= facecam.x or centred >= facecam.right:
        return round(centred)
    options: list[float] = []
    if facecam.x >= width:
        options.append(facecam.x - width)
    if source_w - facecam.right >= width:
        options.append(facecam.right)
    if not options:
        return round(centred)
    nearest = min(options, key=lambda x: abs(x - centred))
    return round(nearest)


def _facecam_hits(segment: CropSegment, tracks: Sequence[FaceTrack], facecam: Rect) -> int:
    return sum(
        1
        for track in tracks
        for o in track.observations_between(segment.start_s, segment.end_s)
        if facecam.x <= o.cx <= facecam.right and facecam.y <= o.cy <= facecam.bottom
    )


def _has_faces(segment: CropSegment, tracks: Sequence[FaceTrack]) -> bool:
    return any(t.observations_between(segment.start_s, segment.end_s) for t in tracks)


def _stack(
    segments: list[CropSegment],
    tracks: Sequence[FaceTrack],
    facecam: Rect,
    usable_h: int,
    source_w: int,
    aspect: float,
) -> list[CropSegment]:
    def stacked(s: CropSegment) -> CropSegment:
        return stacked_segment(
            s.start_s, s.end_s, facecam=facecam, usable_h=usable_h, source_w=source_w, aspect=aspect
        )

    flags = [_facecam_hits(s, tracks, facecam) >= STACK_MIN_FACECAM_HITS for s in segments]
    # Short faceless gaps between stacked segments are facecam dropouts.
    for i in range(1, len(segments) - 1):
        if (
            not flags[i]
            and flags[i - 1]
            and flags[i + 1]
            and segments[i].duration_s <= STACK_GAP_FILL_S
            and not _has_faces(segments[i], tracks)
        ):
            flags[i] = True

    result: list[CropSegment] = []
    for segment, flag in zip(segments, flags, strict=True):
        if not flag:
            result.append(segment)
        elif result and result[-1].strategy is Strategy.STACKED:
            # Same layout either side: one segment, no needless cut.
            result[-1].end_s = segment.end_s
        else:
            result.append(stacked(segment))
    return result


# --------------------------------------------------------------------------
# Subtitles
# --------------------------------------------------------------------------


def _above_subtitles(
    segment: CropSegment, usable_h: int, source_w: int, aspect: float
) -> CropSegment:
    """Keep the segment's window above ``usable_h``.

    A crop too tall to fit shrinks (same aspect, same centre); one that merely
    reaches down moves up. A fitted window is cut at the subtitle line.
    """
    if segment.strategy is Strategy.STACKED:
        return segment  # built against usable_h already

    if segment.fit:
        window = segment.window
        height = _even(min(window.height, usable_h - window.y))
        return replace(segment, height=height)

    width, height = segment.width, segment.height
    if height > usable_h:
        height = usable_h
        width = min(source_w, _even(height * aspect))
    dx = (segment.width - width) / 2
    keyframes = [
        CropKeyframe(
            t=k.t,
            x=_clamp(k.x + dx, 0, source_w - width),
            y=_clamp(k.y, 0, usable_h - height),
        )
        for k in segment.keyframes
    ]
    return replace(segment, width=width, height=height, keyframes=keyframes)


# --------------------------------------------------------------------------
# Opening
# --------------------------------------------------------------------------


def _opening(
    segments: list[CropSegment], usable_h: int, source_w: int, aspect: float
) -> list[CropSegment]:
    """Show the full width over the first :data:`OPENING_S` seconds."""
    result: list[CropSegment] = []
    for segment in segments:
        if segment.start_s >= OPENING_S or segment.strategy is Strategy.TRACK:
            result.append(segment)
            continue
        if segment.start_s > 0 and OPENING_S - segment.start_s < MIN_SEGMENT_S:
            # Too little of the opening left to show without a flash.
            result.append(segment)
            continue

        split = OPENING_S if segment.end_s - OPENING_S >= MIN_SEGMENT_S else segment.end_s
        result.append(_full_width(segment, segment.start_s, split, usable_h, source_w, aspect))
        if split < segment.end_s:
            rest = replace(segment, start_s=split)
            if rest.keyframes:
                # Only the time of the first keyframe moves; the window doesn't.
                rest.keyframes = [replace(rest.keyframes[0], t=split), *rest.keyframes[1:]]
            result.append(rest)
    return result


def _full_width(
    segment: CropSegment,
    start_s: float,
    end_s: float,
    usable_h: int,
    source_w: int,
    aspect: float,
) -> CropSegment:
    # A stacked segment keeps its facecam band; the gameplay below goes full width.
    stacked = segment.strategy is Strategy.STACKED
    return replace(
        segment,
        start_s=start_s,
        end_s=end_s,
        width=source_w,
        height=usable_h,
        keyframes=[CropKeyframe(t=start_s, x=0, y=0)],
        fit=not stacked,
        zoom=0.0,
    )


def _merge(segments: list[CropSegment]) -> list[CropSegment]:
    """Join neighbouring static segments that show the same thing.

    Separate shots often end up with the same centre crop or the same fit; one
    segment renders the same picture with fewer cuts in the filtergraph.
    """
    result: list[CropSegment] = []
    for segment in segments:
        previous = result[-1] if result else None
        if (
            previous is not None
            and previous.is_static
            and segment.is_static
            and previous.window == segment.window
            and (previous.strategy, previous.fit, previous.zoom, previous.stack_top)
            == (segment.strategy, segment.fit, segment.zoom, segment.stack_top)
        ):
            previous.end_s = segment.end_s
        else:
            result.append(replace(segment))
    return result


def _even(value: float) -> int:
    rounded = int(round(value))
    return max(2, rounded - rounded % 2)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(value, high)) if high > low else max(0.0, low)
