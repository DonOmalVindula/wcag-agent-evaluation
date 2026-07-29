"""
Unit tests for the batch orchestrator: checkpoint/resume semantics,
environment loading, and the robots.txt ethics gate. All offline —
network calls are stubbed.
"""

import json

import pytest

from src.agent.batch_runner import load_completed, load_dotenv, robots_allows, run_key


# ---------------------------------------------------------------------------
# Checkpoint / resume
# ---------------------------------------------------------------------------

def _rec(task_id="a.example::info", mode="dom", rep=0, status="success"):
    return {"task_id": task_id, "observation_mode": mode, "run_index": rep,
            "status": status, "error": ""}


class TestCheckpointResume:
    def test_completed_runs_are_skipped(self, tmp_path):
        f = tmp_path / "runs.jsonl"
        f.write_text(json.dumps(_rec()) + "\n")
        done = load_completed(f)
        assert run_key("a.example::info", "dom", 0) in done

    def test_errored_runs_are_retried(self, tmp_path):
        # Infrastructure errors must NOT count as completed on resume
        f = tmp_path / "runs.jsonl"
        f.write_text(json.dumps(_rec(status="error")) + "\n")
        assert load_completed(f) == set()
        assert run_key("a.example::info", "dom", 0) in load_completed(
            f, include_errors=True)

    def test_task_failures_are_not_retried(self, tmp_path):
        # A legitimate task failure IS a result; only errors get re-run
        f = tmp_path / "runs.jsonl"
        f.write_text(json.dumps(_rec(status="failure")) + "\n"
                     + json.dumps(_rec(rep=1, status="timeout")) + "\n")
        done = load_completed(f)
        assert len(done) == 2

    def test_error_then_success_counts_once(self, tmp_path):
        f = tmp_path / "runs.jsonl"
        f.write_text(json.dumps(_rec(status="error")) + "\n"
                     + json.dumps(_rec(status="success")) + "\n")
        assert len(load_completed(f)) == 1

    def test_corrupt_lines_are_tolerated(self, tmp_path):
        f = tmp_path / "runs.jsonl"
        f.write_text('{"broken json\n' + json.dumps(_rec()) + "\n\n")
        assert len(load_completed(f)) == 1

    def test_missing_file_is_empty(self, tmp_path):
        assert load_completed(tmp_path / "nope.jsonl") == set()


# ---------------------------------------------------------------------------
# .env loading
# ---------------------------------------------------------------------------

class TestDotenv:
    def test_loads_key_values(self, tmp_path, monkeypatch):
        import os
        monkeypatch.delenv("TEST_DOTENV_KEY", raising=False)
        f = tmp_path / ".env"
        f.write_text("# comment\nTEST_DOTENV_KEY='secret-value'\n\nBAD LINE\n")
        load_dotenv(f)
        assert os.environ["TEST_DOTENV_KEY"] == "secret-value"
        monkeypatch.delenv("TEST_DOTENV_KEY", raising=False)

    def test_does_not_override_existing(self, tmp_path, monkeypatch):
        import os
        monkeypatch.setenv("TEST_DOTENV_KEY2", "original")
        f = tmp_path / ".env"
        f.write_text("TEST_DOTENV_KEY2=overwritten\n")
        load_dotenv(f)
        assert os.environ["TEST_DOTENV_KEY2"] == "original"

    def test_missing_file_is_noop(self, tmp_path):
        load_dotenv(tmp_path / "absent.env")  # must not raise


# ---------------------------------------------------------------------------
# robots.txt ethics gate (httpx stubbed)
# ---------------------------------------------------------------------------

class _FakeResponse:
    def __init__(self, status_code=200, text="", content_type="text/plain"):
        self.status_code = status_code
        self.text = text
        self.headers = {"content-type": content_type}


def _fake_client(response=None, exc=None):
    class _Client:
        def __init__(self, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url, timeout=None):
            if exc:
                raise exc
            return response

    return _Client


class TestRobotsGate:
    async def test_disallow_all_blocks_site(self, monkeypatch):
        import httpx
        monkeypatch.setattr(httpx, "AsyncClient",
                            _fake_client(_FakeResponse(text="User-agent: *\nDisallow: /\n")))
        assert await robots_allows("https://blocked.example/", {}) is False

    async def test_allow_by_default(self, monkeypatch):
        import httpx
        monkeypatch.setattr(httpx, "AsyncClient",
                            _fake_client(_FakeResponse(text="User-agent: *\nDisallow: /admin\n")))
        assert await robots_allows("https://open.example/", {}) is True

    async def test_missing_robots_txt_allows(self, monkeypatch):
        import httpx
        monkeypatch.setattr(httpx, "AsyncClient",
                            _fake_client(_FakeResponse(status_code=404)))
        assert await robots_allows("https://no-robots.example/", {}) is True

    async def test_network_failure_defaults_to_allowed(self, monkeypatch):
        import httpx
        monkeypatch.setattr(httpx, "AsyncClient",
                            _fake_client(exc=RuntimeError("connection refused")))
        assert await robots_allows("https://flaky.example/", {}) is True

    async def test_result_is_cached_per_origin(self, monkeypatch):
        import httpx
        cache: dict = {}
        monkeypatch.setattr(httpx, "AsyncClient",
                            _fake_client(_FakeResponse(text="User-agent: *\nDisallow: /\n")))
        await robots_allows("https://c.example/page1", cache)
        monkeypatch.setattr(httpx, "AsyncClient",
                            _fake_client(exc=RuntimeError("must not be called")))
        # Second call for the same origin must come from cache, not the network
        assert await robots_allows("https://c.example/page2", cache) is False
