"""Per-dimension clip scores and the hook headline (prompt highlight_v2).

Covers the weighted total, how candidates carry the new fields through to the
database and API, and that databases created before the fields existed still
load their clips.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator

import pytest
from autoclip import app as app_module
from autoclip import db, paths
from autoclip.app import create_app
from autoclip.db import schema, store
from autoclip.db.models import Clip, Job, Source, new_id
from autoclip.pipeline import highlights
from autoclip.pipeline.transcript import Transcript, Word
from autoclip.providers import ClipCandidate, DetectionConfig
from autoclip.providers.base import load_prompt, weighted_score
from fastapi.testclient import TestClient

SUBSCORES = {"score_hook": 8, "score_story": 7, "score_engagement": 7, "score_standalone": 8}


class TestWeightedScore:
    def test_formula(self) -> None:
        # round(10 * (0.35*8 + 0.25*8 + 0.20*7 + 0.20*7)) = 76
        assert weighted_score(8, 7, 7, 8) == 76

    def test_halves_round_up(self) -> None:
        # 10 * 0.35 * 1 = 3.5
        assert weighted_score(1, 0, 0, 0) == 4

    def test_range(self) -> None:
        assert weighted_score(0, 0, 0, 0) == 0
        assert weighted_score(10, 10, 10, 10) == 100

    def test_hook_weighs_most(self) -> None:
        assert (
            weighted_score(10, 0, 0, 0) > weighted_score(0, 0, 0, 10) > weighted_score(0, 10, 0, 0)
        )


class TestCandidate:
    def test_all_subscores_set_the_total(self) -> None:
        candidate = ClipCandidate(start_word_index=0, end_word_index=9, score=99, **SUBSCORES)

        assert candidate.score == 76

    def test_partial_subscores_keep_the_given_total(self) -> None:
        candidate = ClipCandidate(start_word_index=0, end_word_index=9, score=64, score_hook=9)

        assert candidate.score == 64
        assert candidate.score_story is None

    def test_v1_candidate_has_no_subscores(self) -> None:
        candidate = ClipCandidate(start_word_index=0, end_word_index=9, score=70)

        assert candidate.score == 70
        assert candidate.hook_headline == ""

    @pytest.mark.parametrize(
        ("raw", "expected"), [(7.6, 8), ("6", 6), (14, 10), (-2, 0), ("high", None), (None, None)]
    )
    def test_subscores_are_coerced(self, raw, expected) -> None:
        candidate = ClipCandidate(start_word_index=0, end_word_index=9, score_hook=raw)

        assert candidate.score_hook == expected

    def test_headline_null_becomes_empty(self) -> None:
        candidate = ClipCandidate(start_word_index=0, end_word_index=9, hook_headline=None)

        assert candidate.hook_headline == ""


def test_prompt_v2_exists_and_names_every_field() -> None:
    prompt = load_prompt("highlight_v2")

    for name in (*SUBSCORES, "hook_headline", "score"):
        assert f'"{name}"' in prompt


def test_build_clips_carries_subscores_and_headline() -> None:
    words = [Word(f"w{i}.", i * 0.5, i * 0.5 + 0.4) for i in range(100)]
    candidate = ClipCandidate(
        start_word_index=0, end_word_index=59, hook_headline="  The headline ", **SUBSCORES
    )

    (clip,) = highlights.build_clips(
        Transcript(words=words), [candidate], DetectionConfig(), job_id="j"
    )

    assert (clip.score_hook, clip.score_story, clip.score_engagement, clip.score_standalone) == (
        8,
        7,
        7,
        8,
    )
    assert clip.score == 76
    assert clip.hook_headline == "The headline"


@pytest.fixture
def job(initialised_db) -> Job:
    source = store.create_source(
        Source(id=new_id(), type="upload", path="C:/media/v.mp4", duration_s=60.0)
    )
    return store.create_job(Job(id=new_id(), source_id=source.id, provider="external", settings={}))


class TestStorage:
    def test_round_trip(self, job: Job) -> None:
        clip = Clip(
            id=new_id(),
            job_id=job.id,
            start_s=0.0,
            end_s=30.0,
            score=76,
            hook_headline="He studied aviation. Watch him land this.",
            **SUBSCORES,
        )
        store.replace_clips(job.id, [clip])

        (loaded,) = store.list_clips(job.id)

        assert loaded.score_hook == 8
        assert loaded.score_standalone == 8
        assert loaded.hook_headline == "He studied aviation. Watch him land this."

    def test_clip_without_subscores_stores_nulls(self, job: Job) -> None:
        store.replace_clips(job.id, [Clip(id=new_id(), job_id=job.id, start_s=0, end_s=30)])

        (loaded,) = store.list_clips(job.id)

        assert loaded.score_hook is None
        assert loaded.hook_headline == ""


def test_old_database_still_loads_its_clips(autoclip_home) -> None:
    """A v1 database with clips migrates and its clips load with empty new fields."""
    paths.ensure_layout()
    conn = sqlite3.connect(paths.db_path())
    with conn:
        schema._migration_v1(conn)
        conn.execute("PRAGMA user_version = 1")
        conn.execute(
            "INSERT INTO sources (id, type, path, created_at) VALUES ('s', 'upload', 'x', 't')"
        )
        conn.execute(
            "INSERT INTO jobs (id, source_id, status, provider, settings_json, created_at,"
            " updated_at) VALUES ('j', 's', 'done', 'anthropic', '{}', 't', 't')"
        )
        conn.execute(
            "INSERT INTO clips (id, job_id, rank, start_s, end_s, title, score, created_at)"
            " VALUES ('c', 'j', 1, 0, 30, 'Old clip', 81, 't')"
        )
    conn.close()

    assert db.init() == schema.SCHEMA_VERSION

    (clip,) = store.list_clips("j")
    assert (clip.title, clip.score) == ("Old clip", 81)
    assert clip.score_hook is None
    assert clip.hook_headline == ""


@pytest.fixture
def client(autoclip_home, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv(app_module.ENV_NO_WORKER, "1")
    with TestClient(create_app()) as test_client:
        yield test_client


def test_api_returns_subscores_and_headline(client: TestClient) -> None:
    source = store.create_source(
        Source(id=new_id(), type="upload", path="C:/media/v.mp4", duration_s=60.0)
    )
    job = store.create_job(Job(id=new_id(), source_id=source.id, provider="external", settings={}))
    store.replace_clips(
        job.id,
        [
            Clip(
                id=new_id(),
                job_id=job.id,
                start_s=0,
                end_s=30,
                score=76,
                hook_headline="Headline",
                **SUBSCORES,
            ),
            Clip(id=new_id(), job_id=job.id, start_s=40, end_s=70, score=55),
        ],
    )

    first, second = client.get(f"/api/jobs/{job.id}/clips").json()

    assert {k: first[k] for k in SUBSCORES} == SUBSCORES
    assert first["hook_headline"] == "Headline"
    assert second["score_hook"] is None
    assert second["hook_headline"] == ""
