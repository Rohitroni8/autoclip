"""External provider: clip selection handed over as work/{job_id}/selection.json.

Covers the file contract (missing, invalid, valid, empty), the hook check that
catches a judge citing the wrong word index, and the pause-and-resume flow:
the job fails "Awaiting selection" and a rerun resumes at highlights without
transcribing again.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from autoclip.config import Settings
from autoclip.db import store
from autoclip.db.models import Job, Source, new_id
from autoclip.pipeline import highlights, transcribe
from autoclip.pipeline.runner import JobWorkspace, PipelineRunner
from autoclip.pipeline.transcript import Transcript, Word
from autoclip.providers import (
    PROVIDERS,
    DetectionConfig,
    ExternalProvider,
    ProviderError,
    build_provider,
)
from autoclip.providers.external_provider import (
    AWAITING_SELECTION,
    hook_matches,
    selection_path,
)

JOB = "job123"
WORDS_PER_SENTENCE = 10
WORD_S = 0.5


def make_transcript(n_words: int = 200) -> Transcript:
    """Words 0.5 s apart; every tenth word closes a sentence."""
    words = []
    for i in range(n_words):
        text = f"w{i}"
        if (i + 1) % WORDS_PER_SENTENCE == 0:
            text += "."
        words.append(Word(text=text, start=i * WORD_S, end=i * WORD_S + 0.4))
    return Transcript(words=words, language="en", model="test")


def clip(start: int = 0, end: int = 79, **overrides) -> dict:
    data = {
        "start_word_index": start,
        "end_word_index": end,
        "title": "A specific title",
        "hook": f"w{start} w{start + 1} w{start + 2}",
        "hook_headline": "The headline",
        "score_hook": 8,
        "score_story": 7,
        "score_engagement": 7,
        "score_standalone": 8,
        "score": 77,
        "reason": "Opens on tension and pays off.",
    }
    data.update(overrides)
    return data


def write_selection(
    clips: list[dict], *, job_id: str = JOB, named_job: str | None = None, **overrides
) -> Path:
    """Write job_id's selection.json; ``named_job`` puts a different id inside it."""
    payload = {"job_id": named_job or job_id, "prompt_version": "highlight_v2", "clips": clips}
    payload.update(overrides)
    path = selection_path(job_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


@pytest.fixture
def provider() -> ExternalProvider:
    return ExternalProvider("")


@pytest.fixture
def transcript() -> Transcript:
    return make_transcript()


class TestRegistration:
    def test_registered_as_external(self) -> None:
        assert PROVIDERS["external"] is ExternalProvider

    def test_builds_without_an_api_key(self) -> None:
        settings = Settings(active_provider="external")

        assert isinstance(build_provider(settings=settings), ExternalProvider)

    def test_health_check_is_available(self, provider: ExternalProvider) -> None:
        assert asyncio.run(provider.health_check()).available


class TestMissingFile:
    def test_raises_awaiting_selection(self, provider, transcript) -> None:
        with pytest.raises(ProviderError) as info:
            provider.select(transcript, job_id=JOB)

        assert str(info.value).startswith(AWAITING_SELECTION)
        assert "selection.json" in str(info.value)
        assert f"/api/jobs/{JOB}/retry" in str(info.value)


class TestInvalidFile:
    def test_not_json(self, provider, transcript) -> None:
        path = selection_path(JOB)
        path.parent.mkdir(parents=True)
        path.write_text("{not json", encoding="utf-8")

        with pytest.raises(ProviderError, match="invalid"):
            provider.select(transcript, job_id=JOB)

    @pytest.mark.parametrize(
        "bad",
        [
            {"score_hook": 11},
            {"score": 101},
            {"title": "x" * 61},
            {"end_word_index": 0},  # not after start
            {"start_word_index": -1},
            {"unexpected": True},
        ],
        ids=["subscore>10", "score>100", "title>60", "end<=start", "negative", "extra-field"],
    )
    def test_schema_violation(self, provider, transcript, bad) -> None:
        write_selection([clip(**bad)])

        with pytest.raises(ProviderError, match="invalid"):
            provider.select(transcript, job_id=JOB)

    def test_missing_required_field(self, provider, transcript) -> None:
        data = clip()
        del data["hook_headline"]
        write_selection([data])

        with pytest.raises(ProviderError, match="invalid"):
            provider.select(transcript, job_id=JOB)

    def test_wrong_prompt_version(self, provider, transcript) -> None:
        write_selection([clip()], prompt_version="highlight_v1")

        with pytest.raises(ProviderError, match="invalid"):
            provider.select(transcript, job_id=JOB)

    def test_selection_for_another_job(self, provider, transcript) -> None:
        write_selection([clip()], named_job="other")

        with pytest.raises(ProviderError, match="belongs to job other"):
            provider.select(transcript, job_id=JOB)

    def test_index_past_the_transcript(self, provider, transcript) -> None:
        write_selection([clip(start=150, end=200)])

        with pytest.raises(ProviderError) as info:
            provider.select(transcript, job_id=JOB)

        assert "200 words" in str(info.value)


class TestValidFile:
    def test_candidates_carry_the_selection(self, provider, transcript) -> None:
        write_selection([clip(0, 79), clip(100, 179, score=60)])

        candidates = provider.select(transcript, job_id=JOB)

        assert [(c.start_word_index, c.end_word_index) for c in candidates] == [(0, 79), (100, 179)]
        assert candidates[0].score == 77
        assert candidates[0].title == "A specific title"
        assert candidates[0].hook == "w0 w1 w2"

    def test_empty_clip_list_is_a_valid_answer(self, provider, transcript) -> None:
        write_selection([])

        assert provider.select(transcript, job_id=JOB) == []


class TestHookCheck:
    def test_wrong_hook_rejects_only_that_clip(self, provider, transcript) -> None:
        write_selection([clip(0, 79), clip(100, 179, hook="w101 w102")])

        candidates = provider.select(transcript, job_id=JOB)

        assert [c.start_word_index for c in candidates] == [0]

    def test_all_hooks_wrong_is_an_error(self, provider, transcript) -> None:
        write_selection([clip(0, 79, hook="nothing like it")])

        with pytest.raises(ProviderError, match="rejected"):
            provider.select(transcript, job_id=JOB)

    def test_punctuation_and_case_are_ignored(self) -> None:
        t = Transcript(words=[Word("Nobody", 0, 1), Word("tells,", 1, 2), Word("you.", 2, 3)])

        assert hook_matches(t, 0, "nobody tells you")
        assert hook_matches(t, 0, "Nobody tells, you.")

    def test_devanagari_vowel_signs_are_kept(self) -> None:
        t = Transcript(
            words=[Word("मिशन", 0, 1), Word("पर", 1, 2), Word("ध्यान", 2, 3), Word("दो।", 3, 4)]
        )

        assert hook_matches(t, 0, "मिशन पर ध्यान दो")
        # Dropping a vowel sign is a different word, not a punctuation slip.
        assert not hook_matches(t, 0, "मशन पर ध्यान दो")

    def test_empty_hook_never_matches(self) -> None:
        assert not hook_matches(make_transcript(), 0, " , ")


class TestHighlightsDetect:
    def test_skips_windows_and_builds_clips(self, provider, transcript, monkeypatch) -> None:
        def no_windows(*args, **kwargs):
            raise AssertionError("external selection must not be windowed")

        monkeypatch.setattr(highlights, "build_windows", no_windows)
        write_selection([clip(0, 79), clip(100, 179, score=60)])

        clips = asyncio.run(highlights.detect(transcript, provider, DetectionConfig(), job_id=JOB))

        assert [c.rank for c in clips] == [1, 2]
        assert clips[0].score == 77
        assert clips[0].start_word == 0

    def test_empty_selection_yields_no_clips(self, provider, transcript) -> None:
        write_selection([])

        assert (
            asyncio.run(highlights.detect(transcript, provider, DetectionConfig(), job_id=JOB))
            == []
        )


class TestPauseAndResume:
    """A job waits at highlights, then resumes there once selection.json exists."""

    @pytest.fixture
    def job(self, initialised_db, monkeypatch) -> Job:
        source = store.create_source(
            Source(id=new_id(), type="upload", path="unused.mp4", duration_s=100.0, has_video=False)
        )
        job = store.create_job(
            Job(id=new_id(), source_id=source.id, provider="external", settings={})
        )

        # Earlier stages already ran: their artifacts are on disk.
        workspace = JobWorkspace(job.id)
        workspace.audio.write_bytes(b"RIFF")
        make_transcript().save(workspace.transcript)
        workspace.silences.write_text("[]", encoding="utf-8")

        def must_not_transcribe(*args, **kwargs):
            raise AssertionError("transcription must not run again")

        monkeypatch.setattr(transcribe, "transcribe", must_not_transcribe)
        # Rendering is out of scope here; record what would have been exported.
        monkeypatch.setattr(PipelineRunner, "_stage_export", lambda self, clips, *a: None)
        return job

    def run(self, job: Job):
        source = store.get_source(job.source_id)
        fresh = store.get_job(job.id)
        return asyncio.run(PipelineRunner(fresh, source, settings=Settings()).run())

    def test_waits_then_resumes_without_transcribing(self, job: Job) -> None:
        with pytest.raises(ProviderError):
            self.run(job)

        waiting = store.get_job(job.id)
        assert waiting.status == "failed"
        assert waiting.current_stage == "highlights"
        assert waiting.error.startswith(AWAITING_SELECTION)

        write_selection([clip(0, 79)], job_id=job.id)
        clips = self.run(job)

        assert len(clips) == 1
        assert store.get_job(job.id).status == "done"
        assert [c.start_word for c in store.list_clips(job.id)] == [0]
