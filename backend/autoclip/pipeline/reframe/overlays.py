"""Things drawn over the source picture that reframing must account for.

Two overlays matter for vertical clips cut from gameplay and streams:

- **Burned-in subtitles** from the source (a game's own dialogue subtitles, a
  channel's captions). A 9:16 crop slices them into fragments ("Retuâ€¦") that
  then compete with our captions, so every crop is kept above them.
- **A facecam**: a small box of webcam faces composited over gameplay. Cropping
  to the faces loses the gameplay and usually cuts someone in half; cropping to
  the gameplay loses the people. The stacked layout shows both.

Both detectors are pure functions over sampled frames or face observations, so
they can be tested without video.
"""

from __future__ import annotations

import logging
import math
import statistics
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from .croppath import Rect
from .faces import FaceObservation
from .tracker import FaceTrack

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Burned-in subtitles
# --------------------------------------------------------------------------

#: Subtitles are searched for in this horizontal middle band of the bottom of
#: the frame. The edges are left out because HUD elements (minimaps, health
#: bars) live there and would read as text.
SUBTITLE_SEARCH_TOP = 0.70
SUBTITLE_SEARCH_X = (0.12, 0.88)
#: A pixel is part of subtitle text when it is bright and within a few pixels
#: of something dark: white glyphs with an outline or drop shadow.
TEXT_BRIGHT = 185
TEXT_DARK = 70
TEXT_NEIGHBOURHOOD = 5
#: A row holds text when at least this fraction of the searched width is text.
TEXT_ROW_FRACTION = 0.008
#: A text line is at most this tall (two subtitle lines included).
MAX_BLOCK_HEIGHT = 0.12
#: Subtitles must appear in at least this fraction of sampled frames: dialogue
#: comes and goes, but a band seen only once is more likely a sign in the world.
MIN_PRESENCE = 0.12
#: Space kept between the crop's bottom edge and the top of the subtitles.
SUBTITLE_MARGIN = 0.012
#: Hits within this fraction of the frame height count as the same line.
SUBTITLE_LINE_TOLERANCE = 0.025
#: A second subtitle line starts at most this far above the usual one.
SUBTITLE_SECOND_LINE = 0.09


def text_block_top(frame: Any) -> int | None:
    """Top row of a subtitle-like text block in the bottom of ``frame``, or None.

    ``frame`` is an RGB or BGR uint8 array. Text is bright pixels next to dark
    ones (glyphs with an outline or shadow), concentrated in a few rows: bright
    gameplay texture lights up rows everywhere, which the compactness test
    rejects.
    """
    import cv2
    import numpy as np

    height, width = frame.shape[:2]
    top = int(height * SUBTITLE_SEARCH_TOP)
    left, right = (int(width * f) for f in SUBTITLE_SEARCH_X)
    region = frame[top:, left:right]
    gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY) if region.ndim == 3 else region

    kernel = np.ones((TEXT_NEIGHBOURHOOD, TEXT_NEIGHBOURHOOD), np.uint8)
    darkest_nearby = cv2.erode(gray, kernel)
    text = (gray >= TEXT_BRIGHT) & (darkest_nearby <= TEXT_DARK)

    counts = text.sum(axis=1)
    rows = np.flatnonzero(counts >= max(4, TEXT_ROW_FRACTION * (right - left)))
    if rows.size == 0:
        return None

    first, last = int(rows[0]), int(rows[-1])
    block_height = last - first + 1
    if block_height > MAX_BLOCK_HEIGHT * height:
        return None
    # Most of the rows between first and last carry text; scattered hits don't.
    if rows.size < 0.35 * block_height:
        return None
    return top + first


def detect_subtitle_band(frames: Sequence[Any]) -> int | None:
    """The y above which crops should stay, or None when there are no subtitles.

    Preconditions:
        frames are same-sized RGB or BGR uint8 arrays sampled across the clip.
    """
    if not frames:
        return None
    height = frames[0].shape[0]
    tops = [t for t in (text_block_top(f) for f in frames) if t is not None]
    return subtitle_band_from_tops(tops, frame_count=len(frames), height=height)


def subtitle_band_from_tops(tops: Sequence[int], *, frame_count: int, height: int) -> int | None:
    """Turn per-frame text tops into a band, trusting only a recurring height.

    Subtitles come back to the same line every time they appear; lettering in
    the world (a sign, a T-shirt print, a truck) turns up at a different height
    each time. So the band is anchored on the most common text height, and only
    lines just above it (a second subtitle line) extend it.
    """
    if not tops:
        return None
    tolerance = SUBTITLE_LINE_TOLERANCE * height
    # The anchor is the height with the most other hits within tolerance;
    # ties go to the lower line, where subtitles live.
    anchor = max(sorted(tops), key=lambda t: (sum(abs(o - t) <= tolerance for o in tops), t))
    support = [t for t in tops if abs(t - anchor) <= tolerance]
    if len(support) < max(2, MIN_PRESENCE * frame_count):
        return None
    lines = [t for t in tops if anchor - SUBTITLE_SECOND_LINE * height <= t <= anchor + tolerance]
    # Less a margin so outlines and shadows stay out too.
    return max(0, int(min(lines) - SUBTITLE_MARGIN * height))


def sample_frames(video: Path, *, start_s: float, end_s: float, count: int = 24) -> list[Any]:
    """Grab ``count`` evenly spaced frames from the clip, for overlay detection."""
    import cv2

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        return []
    frames: list[Any] = []
    try:
        duration = max(0.0, end_s - start_s)
        for i in range(count):
            t = start_s + duration * (i + 0.5) / count
            capture.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, frame = capture.read()
            if ok:
                frames.append(frame)
    finally:
        capture.release()
    return frames


# --------------------------------------------------------------------------
# Facecam
# --------------------------------------------------------------------------

#: The faces of a facecam together cover at most this much of the frame.
FACECAM_MAX_AREA = 0.30
#: Facecam faces are smallish (a big streamer cam reaches ~25% of the height);
#: a face taller than this is someone on camera full frame, not a webcam box.
FACECAM_MAX_FACE_HEIGHT = 0.33
#: A facecam face is detected in at least this fraction of all frames sampled
#: from the clip. Measured against the whole clip, not against the frames that
#: happen to contain faces: a game character's two-second close-up must not
#: count as a fixture just because the facecam itself is hard to detect.
STABLE_MIN_PRESENCE = 0.20
#: ...and its centre wanders by at most this fraction of the frame (standard
#: deviation). Game characters and people moving through a shot fail this.
STABLE_MAX_DRIFT = 0.04
#: Tracks whose centres are within this many face widths/heights of a seat's
#: centre are the same person in the same facecam spot.
SEAT_TOLERANCE = 0.5
#: Other facecam seats need only this presence, if they sit beside the anchor...
SUPPORT_MIN_PRESENCE = 0.06
#: ...no more than this many face widths from a seat already in the facecam,
#: and on screen at the same time as it in at least this fraction of frames.
SUPPORT_MAX_GAP = 3.5
MIN_TOGETHER = 0.03
#: Padding around the faces, in face widths / heights, so the band shows heads,
#: headsets and shoulders rather than masks of faces.
PAD_SIDE = 0.6
PAD_ABOVE = 0.7
PAD_BELOW = 1.3


def detect_facecam(
    tracks: Sequence[FaceTrack], *, source_w: int, source_h: int, sampled_frames: int
) -> Rect | None:
    """The region where two or more webcam faces sit over other footage, or None.

    A facecam is people who stay put. Each person becomes a *seat*: the
    low-drift face tracks found at the same spot, merged, because the tracker
    splits one person into many short tracks whenever the facecam drops out
    (B-roll, a missed detection). A seat that is present for much of the clip
    is a fixture; faces in the footage underneath (game characters, people
    walking through a shot) come and go or move, so they never form one.
    Full-frame people (a talking head, a green-screen insert) are too big, or
    spread too wide, to pass as an overlay.

    ``sampled_frames`` is how many frames face detection looked at across the
    clip, with or without faces.
    """
    if sampled_frames <= 0:
        return None

    seats = _facecam_cluster(
        _seats(t for t in tracks if _is_steady(t, source_w, source_h)), sampled_frames
    )
    if len(seats) < 2:
        return None

    def quantile(values: list[float], q: float) -> float:
        ordered = sorted(values)
        return ordered[min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))]

    # Robust per-seat bounds: a few mis-detections shouldn't stretch the box.
    left = min(quantile([o.left for o in seat], 0.05) for seat in seats)
    right = max(quantile([o.right for o in seat], 0.95) for seat in seats)
    top = min(quantile([o.top for o in seat], 0.05) for seat in seats)
    bottom = max(quantile([o.bottom for o in seat], 0.95) for seat in seats)

    if (right - left) * (bottom - top) > FACECAM_MAX_AREA * source_w * source_h:
        return None

    observations = [o for seat in seats for o in seat]
    face_w = statistics.median(o.width for o in observations)
    face_h = statistics.median(o.height for o in observations)
    x0 = max(0.0, left - PAD_SIDE * face_w)
    x1 = min(float(source_w), right + PAD_SIDE * face_w)
    y0 = max(0.0, top - PAD_ABOVE * face_h)
    y1 = min(float(source_h), bottom + PAD_BELOW * face_h)
    log.info("Facecam detected at x=%.0f-%.0f y=%.0f-%.0f.", x0, x1, y0, y1)
    return Rect(x0, y0, x1 - x0, y1 - y0)


def _times(seat: list[FaceObservation]) -> set[float]:
    return {round(o.t, 3) for o in seat}


def _presence(seat: list[FaceObservation]) -> int:
    return len(_times(seat))


def _facecam_cluster(
    seats: list[list[FaceObservation]], sampled_frames: int
) -> list[list[FaceObservation]]:
    """The seats that make up the facecam: an anchor and the seats beside it.

    People in a facecam swap places, lean in and out, and join for part of a
    video, so only one of them may hold a spot for long. The anchor is the most
    present seat (at least :data:`STABLE_MIN_PRESENCE`); seats with at least
    :data:`SUPPORT_MIN_PRESENCE` join when they sit at the same height with a
    similar face size, within :data:`SUPPORT_MAX_GAP` face widths of a seat
    already in the cluster, and on screen together with it often enough
    (:data:`MIN_TOGETHER`): a facecam is several people at once, not one
    person who sits in different places.
    """

    def centre(seat: list[FaceObservation]) -> tuple[float, float, float, float]:
        return (
            statistics.median(o.cx for o in seat),
            statistics.median(o.cy for o in seat),
            statistics.median(o.width for o in seat),
            statistics.median(o.height for o in seat),
        )

    ranked = sorted(seats, key=_presence, reverse=True)
    if not ranked or _presence(ranked[0]) < STABLE_MIN_PRESENCE * sampled_frames:
        return []

    cluster = [ranked[0]]
    candidates = [s for s in ranked[1:] if _presence(s) >= SUPPORT_MIN_PRESENCE * sampled_frames]
    grew = True
    while grew:
        grew = False
        for seat in list(candidates):
            cx, cy, w, h = centre(seat)
            for member in cluster:
                mx, my, mw, mh = centre(member)
                together = len(_times(seat) & _times(member))
                if (
                    abs(cy - my) <= 0.5 * mh
                    and 0.7 <= h / mh <= 1.4
                    and abs(cx - mx) <= SUPPORT_MAX_GAP * mw
                    and together >= max(3, MIN_TOGETHER * sampled_frames)
                ):
                    cluster.append(seat)
                    candidates.remove(seat)
                    grew = True
                    break
    return cluster


def _is_steady(track: FaceTrack, source_w: int, source_h: int) -> bool:
    """A track that could be a webcam face: not huge, and barely moving."""
    observations = track.observations
    if len(observations) < 3:
        return False
    if statistics.median(o.height for o in observations) > FACECAM_MAX_FACE_HEIGHT * source_h:
        return False
    return (
        statistics.pstdev(o.cx for o in observations) <= STABLE_MAX_DRIFT * source_w
        and statistics.pstdev(o.cy for o in observations) <= STABLE_MAX_DRIFT * source_h
    )


def _seats(tracks: Iterable[FaceTrack]) -> list[list[FaceObservation]]:
    """Merge steady tracks at the same spot into one observation list per person.

    Same spot means centres within :data:`SEAT_TOLERANCE` of a face size of the
    seat's centre and a similar face size. Longest tracks go first, so a seat's
    centre comes from the best-observed stretch.
    """
    seats: list[list[FaceObservation]] = []
    centres: list[tuple[float, float, float, float]] = []  # cx, cy, width, height
    for track in sorted(tracks, key=lambda t: len(t.observations), reverse=True):
        obs = track.observations
        cx = statistics.median(o.cx for o in obs)
        cy = statistics.median(o.cy for o in obs)
        w = statistics.median(o.width for o in obs)
        h = statistics.median(o.height for o in obs)
        for index, (sx, sy, sw, sh) in enumerate(centres):
            if (
                abs(cx - sx) <= SEAT_TOLERANCE * sw
                and abs(cy - sy) <= SEAT_TOLERANCE * sh
                and 0.7 <= h / sh <= 1.4
            ):
                seats[index].extend(obs)
                break
        else:
            seats.append(list(obs))
            centres.append((cx, cy, w, h))
    return seats


def faces_in(region: Rect, faces: Sequence[FaceObservation]) -> bool:
    """True when there are faces and every one of them is inside ``region``."""
    return bool(faces) and all(
        region.x <= f.cx <= region.right and region.y <= f.cy <= region.bottom for f in faces
    )


def fit_aspect(region: Rect, aspect: float, *, bounds_w: float, bounds_h: float) -> Rect:
    """Grow ``region`` to the given width/height ratio around its centre, inside bounds.

    Growing, never shrinking, keeps every face that was in the region. If the
    bounds can't hold the target ratio, the result is the largest that fits.
    """
    width, height = region.width, region.height
    if width / height < aspect:
        width = height * aspect
    else:
        height = width / aspect
    if width > bounds_w:
        width, height = bounds_w, bounds_w / aspect
    if height > bounds_h:
        width, height = bounds_h * aspect, bounds_h

    # Even sizes for the encoder, rounded up so the region stays inside.
    width = min(_even_up(width), _even_down(bounds_w))
    height = min(_even_up(height), _even_down(bounds_h))
    cx = region.x + region.width / 2
    cy = region.y + region.height / 2
    x = min(max(0.0, cx - width / 2), bounds_w - width)
    y = min(max(0.0, cy - height / 2), bounds_h - height)
    return Rect(math.floor(x), math.floor(y), width, height)


def _even_up(value: float) -> int:
    rounded = math.ceil(value - 1e-6)
    return rounded + rounded % 2


def _even_down(value: float) -> int:
    rounded = math.floor(value + 1e-6)
    return rounded - rounded % 2
