"""Review gate: the pipeline stops at review unless auto_export is on.

Exports are what get posted, so a clip must not be rendered to the exports
folder before someone has looked at it. The review UI previews from the source
and the crop path, so it never needs the pipeline's renders.
"""

from __future__ import annotations

import asyncio

import pytest
from autoclip.config import ExportSettings, Settings
from autoclip.db import store
from autoclip.db.models import Clip, Job, Source, new_id
from autoclip.pipeline.runner import JobWorkspace, PipelineRunner
from autoclip.pipeline.transcript import Transcript, Word


@pytest.fixture
def job(initialised_db) -> Job:
    """A job whose earlier stages are done and whose clips are already detected."""
    source = store.create_source(
        Source(id=new_id(), type="upload", path="unused.mp4", duration_s=60.0, has_video=False)
    )
    job = store.create_job(Job(id=new_id(), source_id=source.id, provider="external", settings={}))

    workspace = JobWorkspace(job.id)
    workspace.audio.write_bytes(b"RIFF")
    words = [Word(text=f"w{i}.", start=i * 0.5, end=i * 0.5 + 0.4) for i in range(100)]
    Transcript(words=words).save(workspace.transcript)
    workspace.silences.write_text("[]", encoding="utf-8")
    store.replace_clips(
        job.id,
        [Clip(id=new_id(), job_id=job.id, start_s=0.0, end_s=30.0, end_word=59, rank=1)],
    )
    return job


@pytest.fixture
def export_calls(monkeypatch) -> list:
    calls: list = []
    monkeypatch.setattr(
        PipelineRunner, "_stage_export", lambda self, clips, *a: calls.append(clips)
    )
    return calls


def run(job: Job, settings: Settings, events: list | None = None):
    runner = PipelineRunner(
        store.get_job(job.id),
        store.get_source(job.source_id),
        settings=settings,
        on_progress=(events.append if events is not None else None),
    )
    return asyncio.run(runner.run())


def test_auto_export_is_off_by_default() -> None:
    assert ExportSettings().auto_export is False


def test_pipeline_stops_at_review(job, export_calls) -> None:
    events: list = []

    clips = run(job, Settings(), events)

    assert export_calls == []
    assert len(clips) == 1
    assert store.list_clips(job.id)[0].status == "candidate"
    done = store.get_job(job.id)
    assert done.status == "done"
    assert done.progress == pytest.approx(1.0)
    assert events[-1].message == "Ready for review"
    assert events[-1].overall == pytest.approx(1.0)


def test_auto_export_restores_the_old_behaviour(job, export_calls) -> None:
    settings = Settings()
    settings.export.auto_export = True

    run(job, settings)

    assert len(export_calls) == 1
    assert [c.rank for c in export_calls[0]] == [1]


def test_old_config_without_the_field_keeps_the_gate() -> None:
    # A config.json written before this setting existed must not start exporting.
    assert Settings.model_validate({"export": {"crf": 20}}).export.auto_export is False
