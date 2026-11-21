"""External provider — clips chosen outside AutoClip, handed over as a file.

The judge (in practice a Claude Code session) reads ``work/{job_id}/transcript.json``,
picks clips by word index, and writes ``work/{job_id}/selection.json``. This
provider only reads and checks that file; it never calls a model and sends
nothing off the machine.

Until the file exists the highlights stage fails with an "Awaiting selection"
error. Every earlier stage has already left its artifacts on disk, so
``POST /api/jobs/{id}/retry`` after the file is written resumes at highlights
without transcribing again.

The file is judged as a whole, not per window: the judge sees the entire
transcript, so the windowing that small-context models need is skipped.
"""

from __future__ import annotations

import json
import logging
import unicodedata
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, model_validator

from .. import paths
from ..pipeline.transcript import Transcript
from .base import (
    ClipCandidate,
    DetectionConfig,
    LLMProvider,
    ProviderError,
    ProviderStatus,
    weighted_score,
)

log = logging.getLogger(__name__)

SELECTION_FILENAME = "selection.json"
PROMPT_VERSION = "highlight_v2"

#: Prefix of the error a job fails with while it waits for a selection. The
#: clip-batch skill matches on it, so treat it as part of the contract.
AWAITING_SELECTION = "Awaiting selection"


class SelectedClip(BaseModel):
    """One clip in ``selection.json`` (handoff contract ``selection_json``)."""

    start_word_index: int = Field(ge=0)
    end_word_index: int = Field(ge=0)
    title: str = Field(max_length=60)
    hook: str
    hook_headline: str
    score_hook: int = Field(ge=0, le=10)
    score_story: int = Field(ge=0, le=10)
    score_engagement: int = Field(ge=0, le=10)
    score_standalone: int = Field(ge=0, le=10)
    score: int = Field(ge=0, le=100)
    reason: str

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _end_after_start(self) -> SelectedClip:
        if self.end_word_index <= self.start_word_index:
            raise ValueError("end_word_index must be greater than start_word_index")
        return self

    @model_validator(mode="after")
    def _score_is_the_weighted_total(self) -> SelectedClip:
        # One point of slack for rounding a half by hand.
        expected = weighted_score(
            self.score_hook, self.score_story, self.score_engagement, self.score_standalone
        )
        if abs(self.score - expected) > 1:
            raise ValueError(
                f"score {self.score} is not the weighted total of the sub-scores "
                f"(expected {expected})"
            )
        return self


class Selection(BaseModel):
    job_id: str
    prompt_version: Literal["highlight_v2"]
    clips: list[SelectedClip]

    model_config = {"extra": "forbid"}


def selection_path(job_id: str) -> Path:
    return paths.job_work_dir(job_id) / SELECTION_FILENAME


def _normalise(text: str) -> list[str]:
    """Lowercase and drop punctuation, keeping letters and combining marks.

    Whisper attaches punctuation to words ("this," / "हैं।") and the judge may
    or may not copy it. Combining marks are kept because Devanagari vowel
    signs are marks, and stripping them would change the word.
    """
    kept = "".join(
        " " if unicodedata.category(ch).startswith(("P", "S")) else ch
        for ch in unicodedata.normalize("NFC", text.casefold())
    )
    return kept.split()


def hook_matches(transcript: Transcript, start: int, hook: str) -> bool:
    """True when ``hook`` is the transcript's wording starting at ``start``."""
    expected = _normalise(hook)
    if not expected:
        return False
    actual: list[str] = []
    for word in transcript.words[start:]:
        actual.extend(_normalise(word.text))
        if len(actual) >= len(expected):
            break
    return actual[: len(expected)] == expected


class ExternalProvider(LLMProvider):
    name = "external"
    requires_key = False

    async def _complete(self, system: str, user: str, config: DetectionConfig) -> str:
        # Selection arrives as a file, so there is no prompt to send. Reaching
        # this means the windowed path was used by mistake.
        raise ProviderError(
            "The external provider does not take prompts.",
            provider=self.name,
            hint="highlights.detect() should call ExternalProvider.select() instead.",
        )

    async def health_check(self) -> ProviderStatus:
        return ProviderStatus(
            name=self.name,
            available=True,
            detail="Reads work/{job_id}/selection.json written by an external judge.",
        )

    def select(self, transcript: Transcript, *, job_id: str) -> list[ClipCandidate]:
        """Read, validate, and convert ``selection.json`` for ``job_id``.

        Raises:
            ProviderError: the file is missing (message starts with
                :data:`AWAITING_SELECTION`), unreadable, fails the schema, or
                every clip in it was rejected.
        """
        path = selection_path(job_id)
        if not path.exists():
            raise ProviderError(
                f"{AWAITING_SELECTION}: no {SELECTION_FILENAME} for job {job_id} yet.",
                provider=self.name,
                hint=(
                    f"Read {path.with_name('transcript.json')}, write {path}, "
                    f"then POST /api/jobs/{job_id}/retry. Transcription is not repeated."
                ),
            )

        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            selection = Selection.model_validate(raw)
        except (OSError, json.JSONDecodeError, ValidationError) as exc:
            raise ProviderError(
                f"{path.name} for job {job_id} is invalid.",
                provider=self.name,
                hint=f"{exc}\n\nFix the file, then retry the job.",
            ) from exc

        if selection.job_id != job_id:
            raise ProviderError(
                f"{path.name} belongs to job {selection.job_id}, not {job_id}.",
                provider=self.name,
            )

        word_count = len(transcript.words)
        out_of_range = [c for c in selection.clips if c.end_word_index >= word_count]
        if out_of_range:
            raise ProviderError(
                f"{path.name} for job {job_id} is invalid.",
                provider=self.name,
                hint=(
                    f"The transcript has {word_count} words (indices 0-{word_count - 1}); "
                    "these clips end past it: "
                    + ", ".join(f"{c.start_word_index}-{c.end_word_index}" for c in out_of_range)
                ),
            )

        candidates: list[ClipCandidate] = []
        rejected: list[str] = []
        for clip in selection.clips:
            if not hook_matches(transcript, clip.start_word_index, clip.hook):
                actual = transcript.text_between(
                    clip.start_word_index,
                    min(clip.start_word_index + 8, word_count - 1),
                )
                rejected.append(
                    f"{clip.start_word_index}: hook {clip.hook!r}, transcript {actual!r}"
                )
                continue
            candidates.append(
                ClipCandidate(
                    start_word_index=clip.start_word_index,
                    end_word_index=clip.end_word_index,
                    title=clip.title,
                    hook=clip.hook,
                    score=clip.score,
                    reason=clip.reason,
                    score_hook=clip.score_hook,
                    score_story=clip.score_story,
                    score_engagement=clip.score_engagement,
                    score_standalone=clip.score_standalone,
                    hook_headline=clip.hook_headline,
                )
            )

        for line in rejected:
            log.warning("Rejected clip starting at word %s", line)

        if rejected and not candidates:
            raise ProviderError(
                f"Every clip in {path.name} for job {job_id} was rejected: "
                "the hook text doesn't match the transcript at start_word_index.",
                provider=self.name,
                hint="\n".join(rejected),
            )

        log.info(
            "Selection for job %s: %d clip(s) accepted, %d rejected.",
            job_id,
            len(candidates),
            len(rejected),
        )
        return candidates
