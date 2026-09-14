"""Tests for notebook agent bootstrap — idempotency, secret resolution, status."""
import threading
import unittest.mock as mock

import pytest

import meshgpu.agent.notebook as nb


def _reset_globals():
    """Reset module-level state between tests."""
    nb._AGENT_THREAD = None
    nb._AGENT_STOP = None
    nb._AGENT_SUPERVISOR = None
    nb._AGENT_STATUS_CALLBACK = None
    nb._AGENT_GENERATION = 0


class TestResolveSecret:
    def test_reads_from_env(self, monkeypatch):
        monkeypatch.setenv("MY_SECRET", "tok_abc")
        assert nb._resolve_secret("MY_SECRET") == "tok_abc"

    def test_returns_none_when_unset(self, monkeypatch):
        monkeypatch.delenv("MY_SECRET", raising=False)
        assert nb._resolve_secret("MY_SECRET") is None

    def test_uses_fallback(self, monkeypatch):
        monkeypatch.delenv("MY_FALLBACK", raising=False)
        assert nb._resolve_secret("MY_FALLBACK", "default_val") == "default_val"


class TestStartIdempotency:
    def setup_method(self):
        _reset_globals()

    def teardown_method(self):
        _reset_globals()

    def test_start_requires_url(self, monkeypatch):
        monkeypatch.delenv("MESHGPU_CONTROLLER", raising=False)
        monkeypatch.delenv("MESHGPU_JOIN_TOKEN", raising=False)
        with pytest.raises(ValueError, match="controller_url"):
            nb.start()

    def test_start_requires_token(self, monkeypatch):
        monkeypatch.setenv("MESHGPU_CONTROLLER", "ws://localhost:8080")
        monkeypatch.delenv("MESHGPU_JOIN_TOKEN", raising=False)
        with pytest.raises(ValueError, match="join_token"):
            nb.start()

    def test_literal_token_is_not_used(self, monkeypatch):
        monkeypatch.setenv("MESHGPU_CONTROLLER", "ws://localhost:8080")
        monkeypatch.delenv("MESHGPU_JOIN_TOKEN", raising=False)
        with pytest.raises(ValueError, match="MESHGPU_JOIN_TOKEN"):
            nb.start(join_token="token-that-must-not-be-used")

    def test_start_is_idempotent(self, monkeypatch):
        """Calling start() twice must not spawn a second thread."""
        monkeypatch.setenv("MESHGPU_CONTROLLER", "ws://localhost:8080")
        monkeypatch.setenv("MESHGPU_JOIN_TOKEN", "tok_xyz")

        alive_thread = mock.MagicMock()
        alive_thread.is_alive.return_value = True
        nb._AGENT_THREAD = alive_thread
        nb._AGENT_SUPERVISOR = mock.MagicMock()
        nb._AGENT_SUPERVISOR._state.worker_id = "w_existing"

        nb.start(controller_url="ws://localhost:8080", join_token="tok_xyz")
        # Should return early without spawning a new thread
        assert nb._AGENT_THREAD is alive_thread

    def test_idempotent_start_updates_status_callback(self, monkeypatch):
        monkeypatch.setenv("MESHGPU_CONTROLLER", "ws://localhost:8080")
        monkeypatch.setenv("MESHGPU_JOIN_TOKEN", "tok_xyz")
        alive_thread = mock.MagicMock()
        alive_thread.is_alive.return_value = True
        nb._AGENT_THREAD = alive_thread
        nb._AGENT_SUPERVISOR = mock.MagicMock()
        nb._AGENT_SUPERVISOR._state.worker_id = "w_existing"
        callback = mock.Mock()

        nb.start(on_status=callback)

        callback.assert_called_once()
        assert callback.call_args.args[0]["worker_id"] == "w_existing"

    def test_start_spawns_thread(self, monkeypatch, tmp_path):
        """
        Full start() path: supervisor is created, thread starts.
        We mock Supervisor.run() so the thread finishes immediately.
        """
        monkeypatch.setenv("MESHGPU_CONTROLLER", "ws://localhost:8080")
        monkeypatch.setenv("MESHGPU_JOIN_TOKEN", "tok_xyz")

        started = threading.Event()

        async def _fake_run(self):
            started.set()

        with mock.patch("meshgpu.agent.supervisor.Supervisor.run", _fake_run):
            nb.start(controller_url="ws://localhost:8080", join_token="tok_xyz")
            assert nb._AGENT_THREAD is not None
            started.wait(timeout=2)

        nb.stop()

    def test_status_callback_receives_start_and_exit(self, monkeypatch):
        monkeypatch.setenv("MESHGPU_CONTROLLER", "ws://localhost:8080")
        monkeypatch.setenv("MESHGPU_JOIN_TOKEN", "tok_xyz")
        finished = threading.Event()
        stopped = threading.Event()
        updates = []

        async def _fake_run(self):
            finished.set()

        def on_status(update):
            updates.append(update)
            if update["running"] is False:
                stopped.set()

        with mock.patch("meshgpu.agent.supervisor.Supervisor.run", _fake_run):
            nb.start(on_status=on_status)
            assert finished.wait(timeout=2)
            assert stopped.wait(timeout=2)
            assert any(update["running"] for update in updates)
            assert any(update["running"] is False for update in updates)

        nb.stop()

    def test_stale_thread_cannot_report_into_replacement_session(self):
        callback = mock.Mock()
        old_supervisor = mock.MagicMock()
        old_supervisor._state.worker_id = "old-worker"
        old_supervisor._state.current_job_id = None
        old_supervisor._state.job_state.value = "STOPPED"
        old_supervisor._state.lease_epoch = 1
        old_supervisor._capability.provider = "old"
        new_supervisor = mock.MagicMock()
        new_supervisor._state.worker_id = "new-worker"
        new_supervisor._state.current_job_id = None
        new_supervisor._state.job_state.value = "RUNNING"
        new_supervisor._state.lease_epoch = 2
        new_supervisor._capability.provider = "new"

        nb._AGENT_GENERATION = 2
        nb._AGENT_SUPERVISOR = new_supervisor
        nb._AGENT_STATUS_CALLBACK = callback
        nb._notify_status(
            running=False,
            generation=1,
            supervisor=old_supervisor,
        )

        callback.assert_not_called()


class TestStatus:
    def setup_method(self):
        _reset_globals()

    def teardown_method(self):
        _reset_globals()

    def test_status_no_supervisor(self):
        assert nb.status() == {"running": False}

    def test_status_with_supervisor(self):
        fake_sup = mock.MagicMock()
        fake_sup._state.worker_id = "w_001"
        fake_sup._state.current_job_id = "job_42"
        fake_sup._state.job_state.value = "RUNNING"
        fake_sup._state.lease_epoch = 3
        fake_sup._capability.provider = "notebook_local_runtime"

        fake_thread = mock.MagicMock()
        fake_thread.is_alive.return_value = True

        nb._AGENT_SUPERVISOR = fake_sup
        nb._AGENT_THREAD = fake_thread

        s = nb.status()
        assert s["running"] is True
        assert s["worker_id"] == "w_001"
        assert s["job_id"] == "job_42"
        assert s["job_state"] == "RUNNING"
        assert s["lease_epoch"] == 3


class TestStop:
    def setup_method(self):
        _reset_globals()

    def teardown_method(self):
        _reset_globals()

    def test_stop_noop_when_not_running(self):
        nb.stop()  # should not raise

    def test_stop_clears_globals(self):
        fake_sup = mock.MagicMock()
        fake_thread = mock.MagicMock()
        fake_thread.join = mock.MagicMock()

        nb._AGENT_SUPERVISOR = fake_sup
        nb._AGENT_THREAD = fake_thread
        nb._AGENT_STOP = threading.Event()

        nb.stop()

        assert nb._AGENT_SUPERVISOR is None
        assert nb._AGENT_THREAD is None
        assert nb._AGENT_STOP is None
        fake_sup.stop.assert_called_once()

    def test_stop_calls_supervisor_stop(self):
        fake_sup = mock.MagicMock()
        fake_thread = mock.MagicMock()
        nb._AGENT_SUPERVISOR = fake_sup
        nb._AGENT_THREAD = fake_thread
        nb._AGENT_STOP = threading.Event()

        nb.stop()
        fake_sup.stop.assert_called_once()
