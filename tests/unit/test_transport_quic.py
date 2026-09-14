"""Tests for QUIC-compat transport abstraction."""
import pytest

from meshgpu.transport.quic_compat import (
    QuicTransport,
    TransportMetrics,
    WebSocketTransport,
    get_transport_class,
    make_transport,
)


class TestGetTransportClass:
    def test_websocket(self):
        assert get_transport_class("websocket") is WebSocketTransport

    def test_quic(self):
        assert get_transport_class("quic") is QuicTransport

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("MESHGPU_TRANSPORT", "quic")
        assert get_transport_class() is QuicTransport

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown transport"):
            get_transport_class("rdma")


class TestTransportMetrics:
    def test_defaults(self):
        m = TransportMetrics()
        assert m.bytes_sent == 0
        assert m.messages_recv == 0


class TestQuicTransportStub:
    def test_not_connected(self):
        qt = QuicTransport("localhost", 4433)
        assert qt.is_connected is False

    @pytest.mark.asyncio
    async def test_connect_raises_not_implemented(self):
        qt = QuicTransport("localhost", 4433)
        with pytest.raises((RuntimeError, NotImplementedError)):
            await qt.connect()

    @pytest.mark.asyncio
    async def test_send_raises_not_implemented(self):
        qt = QuicTransport("localhost", 4433)
        with pytest.raises(NotImplementedError):
            await qt.send(b"hello")

    @pytest.mark.asyncio
    async def test_close_noop(self):
        qt = QuicTransport("localhost", 4433)
        await qt.close()  # should not raise


class TestWebSocketTransportInterface:
    def test_make_transport_returns_ws_transport(self):
        """make_transport with default websocket name wraps a ws."""
        import unittest.mock as mock
        ws = mock.MagicMock()
        ws.closed = False
        t = make_transport(ws, "websocket")
        assert isinstance(t, WebSocketTransport)
        assert t.is_connected is True

    def test_make_transport_bad_name_raises(self):
        import unittest.mock as mock
        ws = mock.MagicMock()
        with pytest.raises((ValueError, RuntimeError)):
            make_transport(ws, "quic")

    def test_is_connected_reflects_ws_state(self):
        import unittest.mock as mock
        ws = mock.MagicMock()
        ws.closed = True
        t = WebSocketTransport(ws)
        assert t.is_connected is False

    @pytest.mark.asyncio
    async def test_close_sets_flag(self):
        import unittest.mock as mock
        ws = mock.MagicMock()
        ws.closed = False
        ws.close = mock.AsyncMock()
        t = WebSocketTransport(ws)
        await t.close()
        assert t._closed is True
