"""Tests for RelayServer — route management, forwarding, GC."""
import asyncio
import ssl

import pytest

from meshgpu.transport.relay import (
    _MAX_FRAME_BYTES,
    _ROUTE_BUFFER_BYTES,
    RelayServer,
    make_relay_url,
    run_outbound_stage,
)


@pytest.mark.asyncio
async def test_wss_outbound_path_builds_default_ssl_context(monkeypatch):
    """The secure client path must not pass ssl=None to websockets."""
    import websockets.asyncio.client

    captured = {}

    async def fake_connect(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        raise RuntimeError("stop after capturing connect arguments")

    monkeypatch.setattr(websockets.asyncio.client, "connect", fake_connect)
    with pytest.raises(ConnectionError, match="exhausted reconnect attempts"):
        await run_outbound_stage(
            object(),
            "wss://relay.example/v1/relay",
            "stage-secret",
            job_id=1,
            stage_id=0,
            max_reconnect_attempts=0,
        )
    assert captured["url"].startswith("wss://relay.example/v1/relay?")
    assert isinstance(captured["ssl"], ssl.SSLContext)


def test_make_relay_url_encodes_route_without_credentials():
    url = make_relay_url(
        "wss://relay.example/v1/relay",
        job_id="job/42",
        stage_id=1,
        role="worker",
        worker_id="kaggle-a",
    )
    assert url == (
        "wss://relay.example/v1/relay?job_id=job%2F42&stage_id=1&role=worker"
        "&worker_id=kaggle-a"
    )
    assert "token" not in url


def test_make_relay_url_rejects_reserved_query_override():
    with pytest.raises(ValueError, match="reserved"):
        make_relay_url(
            "wss://relay.example/v1/relay?role=gateway",
            job_id="job",
            stage_id=0,
            role="worker",
        )


class TestRouteLifecycle:
    @pytest.mark.asyncio
    async def test_register_and_close(self):
        relay = RelayServer()
        key = await relay.register_route("job1", "w_a", "w_b")
        assert relay.route_count() == 1
        assert "job1" in key
        await relay.close_route("job1", "w_a", "w_b")
        assert relay.route_count() == 0

    @pytest.mark.asyncio
    async def test_register_replaces_existing(self):
        relay = RelayServer()
        await relay.register_route("job1", "w_a", "w_b")
        await relay.register_route("job1", "w_a", "w_b")  # replace
        assert relay.route_count() == 1

    @pytest.mark.asyncio
    async def test_close_job_routes(self):
        relay = RelayServer()
        await relay.register_route("job1", "w_a", "w_b")
        await relay.register_route("job1", "w_b", "w_c")
        await relay.register_route("job2", "w_x", "w_y")
        await relay.close_job_routes("job1")
        assert relay.route_count() == 1

    @pytest.mark.asyncio
    async def test_close_nonexistent_route_noop(self):
        relay = RelayServer()
        await relay.close_route("missing", "a", "b")  # should not raise


class TestForwarding:
    @pytest.mark.asyncio
    async def test_forward_and_drain(self):
        relay = RelayServer()
        await relay.register_route("job1", "src", "dst")

        frame = b"hello world"
        ok = await relay.forward("job1", "src", "dst", frame)
        assert ok is True

        received = []

        async def ws_send(data):
            received.append(data)

        await relay.drain_to_worker("job1", "dst", ws_send)
        assert received == [frame]

    @pytest.mark.asyncio
    async def test_forward_returns_false_unknown_route(self):
        relay = RelayServer()
        ok = await relay.forward("bad_job", "src", "dst", b"data")
        assert ok is False

    @pytest.mark.asyncio
    async def test_forward_returns_false_when_buffer_full(self):
        relay = RelayServer()
        await relay.register_route("job1", "src", "dst")

        # Fill up the credit
        route = relay._routes["job1/src->dst"]
        route.buffered_bytes = _ROUTE_BUFFER_BYTES  # simulate full buffer

        ok = await relay.forward("job1", "src", "dst", b"x")
        assert ok is False

    @pytest.mark.asyncio
    async def test_forward_frame_too_large_raises(self):
        relay = RelayServer()
        await relay.register_route("job1", "src", "dst")
        huge = b"x" * (_MAX_FRAME_BYTES + 1)
        with pytest.raises(ValueError, match="max"):
            await relay.forward("job1", "src", "dst", huge)

    @pytest.mark.asyncio
    async def test_drain_empty_queue_noop(self):
        relay = RelayServer()
        await relay.register_route("job1", "src", "dst")

        sent = []

        async def ws_send(d):
            sent.append(d)

        await relay.drain_to_worker("job1", "dst", ws_send)
        assert sent == []

    @pytest.mark.asyncio
    async def test_buffered_bytes_decreases_after_drain(self):
        relay = RelayServer()
        await relay.register_route("job1", "src", "dst")
        frame = b"abc"
        await relay.forward("job1", "src", "dst", frame)

        route = relay._routes["job1/src->dst"]
        assert route.buffered_bytes == len(frame)

        async def ws_send(d):
            pass

        await relay.drain_to_worker("job1", "dst", ws_send)
        assert route.buffered_bytes == 0

    @pytest.mark.asyncio
    async def test_multiple_frames_ordered(self):
        relay = RelayServer()
        await relay.register_route("job1", "src", "dst")
        frames = [b"frame%d" % i for i in range(5)]
        for f in frames:
            await relay.forward("job1", "src", "dst", f)

        received = []

        async def ws_send(d):
            received.append(d)

        await relay.drain_to_worker("job1", "dst", ws_send)
        assert received == frames


class TestGcLoop:
    @pytest.mark.asyncio
    async def test_gc_removes_idle_routes(self):
        relay = RelayServer()
        await relay.register_route("job1", "src", "dst")
        route = relay._routes["job1/src->dst"]
        # Backdate last_activity past the deadline
        route.last_activity = 0.0  # epoch 0 → always idle

        # Patch sleep to avoid waiting 10s
        async def fast_sleep(s):
            raise asyncio.CancelledError

        orig_sleep = asyncio.sleep
        asyncio.sleep = fast_sleep
        try:
            with pytest.raises(asyncio.CancelledError):
                await relay.gc_loop()
        finally:
            asyncio.sleep = orig_sleep

        assert relay.route_count() == 0


class TestStats:
    @pytest.mark.asyncio
    async def test_stats_empty(self):
        relay = RelayServer()
        s = relay.stats()
        assert s["routes"] == 0
        assert s["total_buffered_bytes"] == 0

    @pytest.mark.asyncio
    async def test_stats_after_forward(self):
        relay = RelayServer()
        await relay.register_route("job1", "src", "dst")
        await relay.forward("job1", "src", "dst", b"0" * 100)
        s = relay.stats()
        assert s["routes"] == 1
        assert s["total_buffered_bytes"] == 100
