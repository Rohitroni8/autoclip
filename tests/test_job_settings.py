"""Jobs run with the settings they were created with.

``POST /api/jobs`` stores a snapshot with the per-job overrides applied; the
runner must use it, or overrides such as the transcription language are
silently ignored for jobs run by the queue.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from autoclip import app as app_module
from autoclip import config
from autoclip.app import create_app
from autoclip.config import Settings
from autoclip.db import store
from autoclip.db.models import Job, Source, new_id
from autoclip.pipeline.runner import PipelineRunner, job_settings
from fastapi.testclient import TestClient


@pytest.fixture
def client(autoclip_home, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv(app_module.ENV_NO_WORKER, "1")
    with TestClient(create_app()) as test_client:
        yield test_client


@pytest.fixture
def source(initialised_db) -> Source:
    return store.create_source(
        Source(id=new_id(), type="upload", path="C:/media/video.mp4", duration_s=600.0)
    )


def saved_language(language: str) -> None:
    saved = config.load()
    saved.whisper.language = language
    config.save(saved)


def test_snapshot_wins_over_saved_config(source: Source) -> None:
    saved_language("en")
    snapshot = Settings()
    snapshot.whisper.language = "hi"
    job = Job(id=new_id(), source_id=source.id, provider="external", settings=snapshot.model_dump())

    assert job_settings(job).whisper.language == "hi"
    assert PipelineRunner(job, source).settings.whisper.language == "hi"


def test_job_without_snapshot_uses_saved_config(source: Source) -> None:
    saved_language("en")
    job = Job(id=new_id(), source_id=source.id, provider="external", settings={})

    assert job_settings(job).whisper.language == "en"


def test_explicit_settings_still_override(source: Source) -> None:
    snapshot = Settings()
    snapshot.whisper.language = "hi"
    job = Job(id=new_id(), source_id=source.id, provider="external", settings=snapshot.model_dump())
    explicit = Settings()
    explicit.whisper.language = "fr"

    assert PipelineRunner(job, source, settings=explicit).settings.whisper.language == "fr"


def test_per_job_language_reaches_the_runner(client: TestClient, source: Source) -> None:
    saved_language("")
    ids = {
        lang: client.post(
            "/api/jobs", json={"source_id": source.id, "settings": {"language": lang}}
        ).json()["id"]
        for lang in ("en", "hi")
    }

    for lang, job_id in ids.items():
        job = store.get_job(job_id)
        assert PipelineRunner(job, source).settings.whisper.language == lang
