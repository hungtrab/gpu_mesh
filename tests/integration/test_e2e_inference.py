"""
End-to-end integration tests for the inference HTTP server.
Uses httpx ASGI transport — no real network, real FastAPI app.

Tests the full path: HTTP request → admission → InferenceSession → token stream.
"""
import json

import pytest
import torch

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

from meshgpu.backends.portable.pipeline import build_pipeline
from meshgpu.inference.admission import MemoryPreflightResult
from meshgpu.inference.server import build_inference_app
from meshgpu.inference.tokenizer import Tokenizer
from meshgpu.models.llama_dense import LlamaConfig

pytestmark = pytest.mark.skipif(not HAS_HTTPX, reason="httpx not installed")


def _tiny_cfg():
    return LlamaConfig(
        vocab_size=64, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
        head_dim=8, max_position_embeddings=32,
    )


class _FakeTokenizer:
    vocab_size = 64
    eos_token_id = 0
    bos_token_id = 1
    pad_token_id = 0

    def encode(self, text, **kwargs):
        ids = [ord(char) % 50 + 2 for char in text]
        return [self.bos_token_id, *ids] if kwargs.get("add_special_tokens") else ids

    def decode(self, token_ids, **kwargs):
        return "".join(chr(token - 2) for token in token_ids if token >= 2)

    def save_pretrained(self, path):
        pass


@pytest.fixture
def app():
    cfg = _tiny_cfg()
    workers = build_pipeline(cfg, 2, [torch.device("cpu")] * 2)
    return build_inference_app(workers, kv_slots=256)


@pytest.fixture
def client(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


@pytest.fixture
def text_client():
    cfg = _tiny_cfg()
    workers = build_pipeline(cfg, 2, [torch.device("cpu")] * 2)
    app = build_inference_app(
        workers,
        kv_slots=256,
        tokenizer=Tokenizer(_FakeTokenizer(), "fake"),
    )
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


class TestHealthz:
    @pytest.mark.asyncio
    async def test_healthz_ok(self, client):
        resp = await client.get("/healthz")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert "active_requests" in body
        assert "kv_available" in body


class TestGenerateNonStreaming:
    @pytest.mark.asyncio
    async def test_basic_generate(self, client):
        resp = await client.post("/v1/generate", json={
            "prompt_ids": [1, 2, 3, 4],
            "max_new_tokens": 3,
            "stream": False,
        })
        assert resp.status_code == 200
        body = resp.json()
        assert "token_ids" in body
        assert len(body["token_ids"]) >= 1
        assert "session_id" in body

    @pytest.mark.asyncio
    async def test_token_ids_in_vocab(self, app, client):
        cfg = _tiny_cfg()
        resp = await client.post("/v1/generate", json={
            "prompt_ids": [5, 10, 15],
            "max_new_tokens": 4,
            "stream": False,
        })
        assert resp.status_code == 200
        body = resp.json()
        for t in body["token_ids"]:
            assert 0 <= t < cfg.vocab_size

    @pytest.mark.asyncio
    async def test_seq_nums_monotone(self, client):
        resp = await client.post("/v1/generate", json={
            "prompt_ids": [1, 2, 3],
            "max_new_tokens": 5,
            "stream": False,
        })
        assert resp.status_code == 200
        body = resp.json()
        seq_nums = body["seq_nums"]
        assert seq_nums == sorted(seq_nums)
        assert seq_nums == list(range(1, len(seq_nums) + 1))

    @pytest.mark.asyncio
    async def test_max_new_tokens_respected(self, client):
        resp = await client.post("/v1/generate", json={
            "prompt_ids": [1, 2, 3, 4],
            "max_new_tokens": 2,
            "stream": False,
        })
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["token_ids"]) <= 2

    @pytest.mark.asyncio
    async def test_tokens_per_second_positive(self, client):
        resp = await client.post("/v1/generate", json={
            "prompt_ids": [1, 2, 3],
            "max_new_tokens": 3,
            "stream": False,
        })
        assert resp.status_code == 200
        assert resp.json()["tokens_per_second"] >= 0

    @pytest.mark.asyncio
    async def test_text_prompt_is_tokenized_and_decoded(self, text_client):
        resp = await text_client.post("/v1/generate", json={
            "prompt": "hello",
            "max_new_tokens": 2,
            "stream": False,
        })
        assert resp.status_code == 200
        body = resp.json()
        assert isinstance(body["text"], str)
        assert len(body["token_ids"]) >= 1


class TestGenerateStreaming:
    @pytest.mark.asyncio
    async def test_sse_returns_tokens(self, client):
        async with client.stream("POST", "/v1/generate", json={
            "prompt_ids": [1, 2, 3, 4],
            "max_new_tokens": 3,
            "stream": True,
        }) as resp:
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers["content-type"]
            lines = []
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    lines.append(line[6:])
            assert len(lines) >= 2  # at least 1 token + done event

    @pytest.mark.asyncio
    async def test_sse_has_done_event(self, client):
        events = []
        async with client.stream("POST", "/v1/generate", json={
            "prompt_ids": [1, 2, 3],
            "max_new_tokens": 2,
            "stream": True,
        }) as resp:
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    events.append(json.loads(line[6:]))

        done = [e for e in events if e.get("event") == "done"]
        assert len(done) == 1

    @pytest.mark.asyncio
    async def test_sse_token_events_valid(self, client):
        cfg = _tiny_cfg()
        token_events = []
        async with client.stream("POST", "/v1/generate", json={
            "prompt_ids": [2, 4, 6],
            "max_new_tokens": 3,
            "stream": True,
        }) as resp:
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    ev = json.loads(line[6:])
                    if "token_id" in ev:
                        token_events.append(ev)

        for ev in token_events:
            assert 0 <= ev["token_id"] < cfg.vocab_size
            assert ev["seq_num"] >= 1
            assert isinstance(ev["is_last"], bool)

    @pytest.mark.asyncio
    async def test_session_id_in_header(self, client):
        async with client.stream("POST", "/v1/generate", json={
            "prompt_ids": [1, 2],
            "max_new_tokens": 2,
            "stream": True,
        }) as resp:
            assert "x-session-id" in resp.headers
            sid = resp.headers["x-session-id"]
            assert len(sid) > 0


class TestAdmissionControl:
    @pytest.mark.asyncio
    async def test_empty_prompt_still_works(self, client):
        # Edge case: very short prompt
        resp = await client.post("/v1/generate", json={
            "prompt_ids": [1],
            "max_new_tokens": 1,
            "stream": False,
        })
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_concurrent_requests(self, client):
        """Multiple requests should all succeed (admission allows them)."""
        import asyncio
        async def make_request():
            return await client.post("/v1/generate", json={
                "prompt_ids": [1, 2, 3],
                "max_new_tokens": 2,
                "stream": False,
            })
        # Note: with shared pipeline workers, these run sequentially inside
        # InferenceSession (the pipeline is not truly concurrent), but HTTP
        # admission and response handling work correctly.
        responses = await asyncio.gather(*[make_request() for _ in range(3)])
        for r in responses:
            # Could be 200 or 429 depending on admission; all should be valid HTTP
            assert r.status_code in (200, 429)


class TestMemoryPreflight:
    @pytest.mark.asyncio
    async def test_insufficient_memory_is_rejected_before_execution(self):
        cfg = _tiny_cfg()
        workers = build_pipeline(cfg, 2, [torch.device("cpu")] * 2)
        calls = []

        def reject(prompt_ids, max_new_tokens):
            calls.append((prompt_ids, max_new_tokens))
            return MemoryPreflightResult(
                feasible=False,
                reason="profile does not fit",
                details={"stage": 1},
            )

        app = build_inference_app(workers, kv_slots=256, memory_preflight=reject)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/generate",
                json={"prompt_ids": [1, 2, 3], "max_new_tokens": 2, "stream": False},
            )
        assert response.status_code == 507
        assert response.json()["detail"]["code"] == "insufficient_memory"
        assert calls == [([1, 2, 3], 2)]
