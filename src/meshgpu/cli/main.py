"""meshgpu CLI entry point."""
from __future__ import annotations

import asyncio
import json
import ssl
import sys
from pathlib import Path

import click


@click.group()
def cli() -> None:
    """MeshGPU — distributed GPU runtime."""


# ------------------------------------------------------------------
# doctor
# ------------------------------------------------------------------

@cli.command()
def doctor() -> None:
    """Check local environment and GPU capabilities."""
    from meshgpu.agent.probe import build_capability_report
    cap = build_capability_report()
    click.echo(json.dumps(cap.model_dump(mode="json"), indent=2))


# ------------------------------------------------------------------
# controller
# ------------------------------------------------------------------

@cli.command()
@click.option("--host", default="0.0.0.0", show_default=True)
@click.option("--port", default=8080, show_default=True)
@click.option("--db", default="meshgpu.db", show_default=True)
@click.option(
    "--join-token",
    default=None,
    envvar="MESHGPU_JOIN_TOKEN",
    help="Pre-shared worker registration token (required for non-dev deployments).",
)
def controller(host: str, port: int, db: str, join_token: str | None) -> None:
    """Start the controller server."""
    import uvicorn

    from meshgpu.controller.server import build_app

    app = build_app(db_path=db, join_token=join_token)
    uvicorn.run(app, host=host, port=port)


# ------------------------------------------------------------------
# relay-server — outbound WebSocket rendezvous for restricted workers
# ------------------------------------------------------------------

@cli.command("relay-server")
@click.option("--host", default="0.0.0.0", show_default=True)
@click.option("--port", default=8443, show_default=True, type=click.IntRange(1, 65535))
@click.option(
    "--relay-token",
    required=True,
    envvar="MESHGPU_RELAY_TOKEN",
    hide_input=True,
    help="Shared relay admission token; use a secret manager in production.",
)
@click.option(
    "--tls-cert",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="PEM server certificate; required for public WSS deployment.",
)
@click.option(
    "--tls-key",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="PEM private key matching --tls-cert.",
)
@click.option("--path", default="/v1/relay", show_default=True)
def relay_server(
    host: str,
    port: int,
    relay_token: str,
    tls_cert: Path | None,
    tls_key: Path | None,
    path: str,
) -> None:
    """Start the trusted relay that pairs outbound worker/gateway sockets."""
    from meshgpu.transport.relay import WebSocketRelay

    try:
        ssl_context = _build_stage_ssl_context(tls_cert, tls_key)
        relay = WebSocketRelay(relay_token=relay_token, path=path)
    except (OSError, ssl.SSLError, ValueError) as exc:
        raise click.ClickException(f"could not prepare relay server: {exc}") from exc

    async def serve_forever() -> None:
        websocket_server = await relay.serve(host, port, ssl_context=ssl_context)
        scheme = "wss" if ssl_context is not None else "ws"
        click.echo(f"MeshGPU relay → {scheme}://{host}:{port}{path}")
        if ssl_context is None:
            click.echo(
                "WARNING: relay TLS is disabled; use wss:// behind a trusted TLS proxy.",
                err=True,
            )
        try:
            await asyncio.Future()
        finally:
            websocket_server.close()
            await websocket_server.wait_closed()
            await relay.close()

    try:
        asyncio.run(serve_forever())
    except KeyboardInterrupt:
        click.echo("MeshGPU relay stopped.")


# ------------------------------------------------------------------
# agent
# ------------------------------------------------------------------

@cli.command()
@click.option("--controller", "controller_url", required=True, envvar="MESHGPU_CONTROLLER")
@click.option("--token", required=True, envvar="MESHGPU_JOIN_TOKEN", help="Join token")
@click.option("--worker-id", default=None, envvar="MESHGPU_WORKER_ID")
@click.option("--provider", default="self_managed", show_default=True)
def agent(controller_url: str, token: str, worker_id: str | None, provider: str) -> None:
    """Start the agent and connect to a controller."""
    import logging

    from meshgpu.agent.supervisor import Supervisor
    logging.basicConfig(level=logging.INFO)

    sup = Supervisor(controller_url, token, worker_id=worker_id, provider=provider)
    try:
        asyncio.run(sup.run())
    except KeyboardInterrupt:
        sup.stop()


# ------------------------------------------------------------------
# workers
# ------------------------------------------------------------------

@cli.group()
def workers() -> None:
    """Worker management commands."""


@workers.command("list")
@click.option("--controller", "controller_url", required=True, envvar="MESHGPU_CONTROLLER")
def workers_list(controller_url: str) -> None:
    """List registered workers."""
    import urllib.request
    with urllib.request.urlopen(f"{controller_url.rstrip('/')}/v1/workers") as resp:
        data = json.loads(resp.read())
    click.echo(json.dumps(data, indent=2))


# ------------------------------------------------------------------
# jobs
# ------------------------------------------------------------------

@cli.group()
def jobs() -> None:
    """Job management commands."""


@jobs.command("status")
@click.argument("job_id")
@click.option("--controller", "controller_url", required=True, envvar="MESHGPU_CONTROLLER")
def jobs_status(job_id: str, controller_url: str) -> None:
    """Get job status."""
    import urllib.request
    url = f"{controller_url.rstrip('/')}/v1/jobs/{job_id}"
    with urllib.request.urlopen(url) as resp:
        data = json.loads(resp.read())
    click.echo(json.dumps(data, indent=2))


@jobs.command("cancel")
@click.argument("job_id")
@click.option("--controller", "controller_url", required=True, envvar="MESHGPU_CONTROLLER")
@click.option("--reason", default="user_cancel")
def jobs_cancel(job_id: str, controller_url: str, reason: str) -> None:
    """Cancel a running job."""
    import urllib.request
    url = f"{controller_url.rstrip('/')}/v1/jobs/{job_id}/cancel"
    body = json.dumps({"reason": reason}).encode()
    req = urllib.request.Request(url, data=body, method="POST",
                                  headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read())
    click.echo(json.dumps(data, indent=2))


# ------------------------------------------------------------------
# plan
# ------------------------------------------------------------------

@cli.command()
@click.argument("job_yaml")
def plan(job_yaml: str) -> None:
    """Estimate feasibility and memory budget for a job spec (YAML)."""
    try:
        import yaml  # type: ignore[import]
    except ImportError:
        click.echo("pyyaml not installed; run: pip install pyyaml", err=True)
        sys.exit(1)

    try:
        from pathlib import Path

        with open(job_yaml, encoding="utf-8") as f:
            spec = yaml.safe_load(f)
        from meshgpu.planner.job import plan_job, report_to_dict

        report = plan_job(spec, base_dir=Path(job_yaml).parent)
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as exc:
        raise click.ClickException(f"could not build placement plan: {exc}") from exc
    click.echo(json.dumps(report_to_dict(report), indent=2))


# ------------------------------------------------------------------
# convert — split HF model into MeshGPU pipeline shards
# ------------------------------------------------------------------

@cli.command()
@click.argument("model_id")
@click.option(
    "--stages", default=2, show_default=True, type=click.IntRange(min=1),
    help="Number of pipeline stages",
)
@click.option("--dtype", default="float16", show_default=True,
              type=click.Choice(["float16", "bfloat16", "float32"]))
@click.option(
    "--revision",
    default=None,
    help="HuggingFace branch, tag, or commit to pin (recommended for reproducible artifacts).",
)
@click.option(
    "--attn-implementation",
    type=click.Choice(["eager", "sdpa", "flash_attention_2"]),
    default="sdpa",
    show_default=True,
    help="Attention backend recorded in each stage artifact (Qwen3 supports flash_attention_2).",
)
@click.option("--out", required=True, help="Output directory for shards + manifest")
@click.option(
    "--include-tokenizer/--no-tokenizer",
    default=True,
    show_default=True,
    help="Copy the HuggingFace tokenizer into the artifact for text inference.",
)
def convert(
    model_id: str,
    stages: int,
    dtype: str,
    revision: str | None,
    attn_implementation: str,
    out: str,
    include_tokenizer: bool,
) -> None:
    """Split a supported HuggingFace decoder model into MeshGPU pipeline shards.

    MODEL_ID can be a HuggingFace hub ID (for example
    TinyLlama/TinyLlama-1.1B-Chat-v1.0, which uses unscaled RoPE)
    or a local directory containing a HF model.

    Example:
        meshgpu convert TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
            --stages 4 --out ./tinyllama-4s
    """
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from meshgpu.artifacts.hf_import import import_from_hf
    click.echo(f"Converting {model_id!r} → {stages} stages → {out}")
    manifest = import_from_hf(
        model_id,
        num_stages=stages,
        out_dir=out,
        dtype=dtype,
        revision=revision,
        include_tokenizer=include_tokenizer,
        attn_implementation=attn_implementation,
    )
    click.echo(f"Done. Manifest: {out}/manifest.json")
    click.echo(f"  {len(manifest.shards)} shards, {manifest.meta.num_layers} layers, "
               f"hidden={manifest.meta.hidden_size}")


# ------------------------------------------------------------------
# serve (local inference — for testing without full controller stack)
# ------------------------------------------------------------------

@cli.command()
@click.option("--host", default="0.0.0.0", show_default=True)
@click.option("--port", default=8090, show_default=True, type=click.IntRange(1, 65535))
@click.option("--manifest", "manifest_dir", default=None,
              help="Directory with manifest.json + shard files (from meshgpu convert).")
@click.option("--num-stages", default=1, show_default=True, type=click.IntRange(min=1),
              help="Stages for the built-in tiny model (ignored when --manifest given).")
@click.option("--device", default="cpu", show_default=True,
              help="Device for all stages: cpu, cuda, cuda:0 …")
@click.option(
    "--devices",
    default=None,
    help="Comma-separated device per stage, e.g. cuda:0,cuda:1 (overrides --device).",
)
@click.option(
    "--transport",
    type=click.Choice(["cpu", "local_cuda"]),
    default="local_cuda",
    show_default=True,
    help="Stage-boundary transport; local_cuda keeps same-host CUDA tensors device-resident.",
)
@click.option("--kv-slots", default=4096, show_default=True, type=click.IntRange(min=1))
@click.option(
    "--max-prompt-tokens",
    default=2048,
    show_default=True,
    type=click.IntRange(min=1),
    help="Maximum prompt length reserved by admission and preflight.",
)
@click.option(
    "--max-new-tokens",
    default=256,
    show_default=True,
    type=click.IntRange(min=1),
    help="Maximum generated tokens reserved by admission and preflight.",
)
@click.option(
    "--max-concurrent",
    default=4,
    show_default=True,
    type=click.IntRange(min=1),
    help="Maximum concurrent requests reserved by admission and preflight.",
)
@click.option(
    "--tokenizer",
    "tokenizer_path",
    default=None,
    help="Tokenizer model/directory; defaults to tokenizer saved in --manifest.",
)
def serve(host: str, port: int, manifest_dir: str | None,
          num_stages: int, device: str, devices: str | None, kv_slots: int,
          transport: str, max_prompt_tokens: int, max_new_tokens: int,
          max_concurrent: int, tokenizer_path: str | None) -> None:
    """Start a local inference server (no controller required).

    With --manifest: loads real weights from a converted model.
    Without --manifest: starts a tiny random-weight model for testing.

    Example (real model):
        meshgpu serve --manifest ./tinyllama-4s --port 8090

    Example (testing):
        meshgpu serve --num-stages 2 --port 8090
    """
    import torch
    import uvicorn

    from meshgpu.inference.admission import AdmissionConfig
    from meshgpu.inference.server import build_inference_app
    from meshgpu.inference.tokenizer import load_tokenizer

    tokenizer = None

    if manifest_dir is not None:
        from pathlib import Path

        from meshgpu.artifacts.manifest import ModelManifest
        from meshgpu.backends.portable.pipeline import build_pipeline_from_manifest
        from meshgpu.planner.preflight import plan_cuda_inference_from_manifest
        manifest = ModelManifest.load(Path(manifest_dir) / "manifest.json")
        n = len(manifest.shards)
        stage_devices = _parse_stage_devices(devices, device, n, torch)
        planned_layer_ranges = None
        has_cuda = any(stage_device.type == "cuda" for stage_device in stage_devices)
        if has_cuda and not all(stage_device.type == "cuda" for stage_device in stage_devices):
            raise click.ClickException(
                "mixed CPU/CUDA inference stages are not supported by automatic "
                "capacity preflight; use all CPU or one CUDA device per stage"
            )
        if has_cuda:
            try:
                preflight = plan_cuda_inference_from_manifest(
                    manifest_dir,
                    stage_devices,
                    max_prompt_tokens=max_prompt_tokens,
                    max_new_tokens=max_new_tokens,
                    max_concurrent=max_concurrent,
                )
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                raise click.ClickException(
                    f"CUDA inference preflight failed before loading weights: {exc}"
                ) from exc
            click.echo("CUDA inference preflight:")
            click.echo(preflight.summary())
            if not preflight.feasible:
                raise click.ClickException(
                    "inference rejected before loading weights: " + preflight.reason
                )
            planned_layer_ranges = [
                (assignment.layer_start, assignment.layer_end)
                for assignment in preflight.assignments
            ]
        click.echo(f"Loading {n}-stage model from {manifest_dir!r} …")
        workers = build_pipeline_from_manifest(
            manifest_dir,
            devices=stage_devices,
            transport=transport,
            layer_ranges=planned_layer_ranges,
        )
        if tokenizer_path is not None:
            tokenizer = load_tokenizer(tokenizer_path, local_files_only=True)
        elif manifest.tokenizer_path is not None:
            from meshgpu.inference.tokenizer import load_tokenizer_from_artifact
            tokenizer = load_tokenizer_from_artifact(manifest_dir, manifest.tokenizer_path)
        click.echo(f"Model: {manifest.meta.model_id}  "
                   f"layers={manifest.meta.num_layers}  "
                   f"hidden={manifest.meta.hidden_size}")
    else:
        from meshgpu.backends.portable.pipeline import build_pipeline
        from meshgpu.models.llama_dense import LlamaConfig
        click.echo(f"No --manifest given; starting tiny model ({num_stages} stage(s)) …")
        cfg = LlamaConfig(
            vocab_size=32000, hidden_size=512, intermediate_size=1024,
            num_hidden_layers=4, num_attention_heads=8, num_key_value_heads=4,
            head_dim=64, max_position_embeddings=2048,
        )
        stage_devices = _parse_stage_devices(devices, device, num_stages, torch)
        workers = build_pipeline(cfg, num_stages, stage_devices, transport=transport)

    admission_cfg = AdmissionConfig(
        max_prompt_tokens=max_prompt_tokens,
        max_new_tokens=max_new_tokens,
        max_concurrent_requests=max_concurrent,
        kv_slots=kv_slots,
    )
    app = build_inference_app(
        workers,
        admission_cfg=admission_cfg,
        kv_slots=kv_slots,
        tokenizer=tokenizer,
    )
    click.echo(f"Inference server → http://{host}:{port}/v1/generate")
    if tokenizer is None:
        click.echo("Text prompts disabled; send prompt_ids or provide --tokenizer.")
    uvicorn.run(app, host=host, port=port)


# ------------------------------------------------------------------
# stage-worker — outbound one-shard worker for relay-restricted hosts
# ------------------------------------------------------------------

@cli.command("stage-worker")
@click.option(
    "--manifest",
    "manifest_dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Artifact directory containing manifest.json and this stage shard.",
)
@click.option(
    "--stage",
    "stage_id",
    required=True,
    type=click.IntRange(min=0),
    help="Zero-based stage/shard index served by this process.",
)
@click.option(
    "--relay-url",
    required=True,
    envvar="MESHGPU_RELAY_URL",
    help="Public relay base URL, e.g. wss://relay.example/v1/relay.",
)
@click.option(
    "--relay-token",
    required=True,
    envvar="MESHGPU_RELAY_TOKEN",
    hide_input=True,
    help="Relay admission token (sent in a header, never in the URL).",
)
@click.option(
    "--credential",
    required=True,
    envvar="MESHGPU_STAGE_CREDENTIAL",
    hide_input=True,
    help="Controller-issued stage credential.",
)
@click.option("--device", default="cuda:0", show_default=True)
@click.option("--cluster-id", required=True, type=click.IntRange(0, 65535))
@click.option("--job-id", required=True, type=click.IntRange(0, 0xFFFFFFFF))
@click.option("--lease-epoch", required=True, type=click.IntRange(0, 0xFFFFFFFF))
@click.option(
    "--worker-incarnation",
    required=True,
    type=click.IntRange(0, 0xFFFFFFFF),
)
@click.option(
    "--peer-worker-incarnation",
    default=None,
    type=click.IntRange(0, 0xFFFFFFFF),
)
@click.option(
    "--transport",
    type=click.Choice(["cpu"]),
    default="cpu",
    show_default=True,
    help="Remote boundaries always serialize through CPU/network buffers.",
)
@click.option(
    "--max-reconnect-attempts",
    default=10,
    show_default=True,
    type=click.IntRange(min=0),
    help="Bounded retries after relay disconnects.",
)
def stage_worker(
    manifest_dir: Path,
    stage_id: int,
    relay_url: str,
    relay_token: str,
    credential: str,
    device: str,
    cluster_id: int,
    job_id: int,
    lease_epoch: int,
    worker_incarnation: int,
    peer_worker_incarnation: int | None,
    transport: str,
    max_reconnect_attempts: int,
) -> None:
    """Load one shard and connect outbound to a public relay.

    This command is intended for Kaggle or another host that cannot expose an
    inbound WebSocket port. The gateway uses the matching ``role=gateway``
    relay URL for this job and stage.
    """
    import logging

    import torch

    from meshgpu.artifacts.hf_import import build_stage_from_manifest
    from meshgpu.backends.portable.rpc import StageRpcIdentity, StageRpcServer
    from meshgpu.transport.relay import run_outbound_stage

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if not credential:
        raise click.ClickException("--credential must not be empty")
    if not relay_token:
        raise click.ClickException("--relay-token must not be empty")
    try:
        stage_device = _parse_stage_devices(None, device, 1, torch)[0]
        worker = build_stage_from_manifest(
            manifest_dir,
            stage_id,
            device=stage_device,
            job_id=str(job_id),
            transport=transport,
        )
        endpoint = StageRpcServer(
            worker,
            StageRpcIdentity(
                cluster_id=cluster_id,
                job_id=job_id,
                lease_epoch=lease_epoch,
                worker_incarnation=worker_incarnation,
                peer_worker_incarnation=peer_worker_incarnation,
            ),
            credential=credential,
        )
        click.echo(
            f"Stage {stage_id} loaded on {stage_device}; connecting outbound to relay"
        )
        asyncio.run(
            run_outbound_stage(
                endpoint,
                relay_url,
                credential,
                job_id=job_id,
                stage_id=stage_id,
                relay_token=relay_token,
                max_reconnect_attempts=max_reconnect_attempts,
            )
        )
    except KeyboardInterrupt:
        click.echo("Outbound stage worker stopped.")
    except (FileNotFoundError, IndexError, KeyError, OSError, RuntimeError, ValueError) as exc:
        raise click.ClickException(f"outbound stage worker failed: {exc}") from exc


# ------------------------------------------------------------------
# stage-server — one authenticated RPC endpoint per model shard
# ------------------------------------------------------------------

@cli.command("stage-server")
@click.option(
    "--manifest",
    "manifest_dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Artifact directory containing manifest.json, its referenced config and this stage shard.",
)
@click.option(
    "--stage",
    "stage_id",
    required=True,
    type=click.IntRange(min=0),
    help="Zero-based stage/shard index served by this process.",
)
@click.option("--host", default="0.0.0.0", show_default=True)
@click.option("--port", default=9100, show_default=True, type=click.IntRange(1, 65535))
@click.option("--device", default="cuda:0", show_default=True)
@click.option(
    "--credential",
    required=True,
    envvar="MESHGPU_STAGE_CREDENTIAL",
    hide_input=True,
    help="Controller-issued stage credential (or MESHGPU_STAGE_CREDENTIAL).",
)
@click.option("--cluster-id", required=True, type=click.IntRange(0, 65535))
@click.option("--job-id", required=True, type=click.IntRange(0, 0xFFFFFFFF))
@click.option("--lease-epoch", required=True, type=click.IntRange(0, 0xFFFFFFFF))
@click.option(
    "--worker-incarnation",
    required=True,
    type=click.IntRange(0, 0xFFFFFFFF),
    help="Numeric incarnation used in tensor frame fencing.",
)
@click.option(
    "--peer-worker-incarnation",
    default=None,
    type=click.IntRange(0, 0xFFFFFFFF),
    help="Optionally fence incoming frames to one known client incarnation.",
)
@click.option(
    "--tls-cert",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="PEM server certificate. Use together with --tls-key outside a trusted LAN.",
)
@click.option(
    "--tls-key",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="PEM private key matching --tls-cert.",
)
@click.option(
    "--transport",
    type=click.Choice(["cpu", "local_cuda"]),
    default="cpu",
    show_default=True,
    help="Boundary transport for the local stage process.",
)
def stage_server(
    manifest_dir: Path,
    stage_id: int,
    host: str,
    port: int,
    device: str,
    credential: str,
    cluster_id: int,
    job_id: int,
    lease_epoch: int,
    worker_incarnation: int,
    peer_worker_incarnation: int | None,
    tls_cert: Path | None,
    tls_key: Path | None,
    transport: str,
) -> None:
    """Serve exactly one converted model shard over authenticated WebSocket RPC.

    Run one instance per stage/GPU.  The artifact directory may contain all
    shard metadata, but this command loads only the selected shard's weights.
    The client must use the matching numeric identity and credential.
    """
    import torch

    from meshgpu.artifacts.hf_import import build_stage_from_manifest
    from meshgpu.backends.portable.rpc import StageRpcIdentity, StageRpcServer

    if not credential:
        raise click.ClickException("--credential must not be empty")
    stage_device = _parse_stage_devices(None, device, 1, torch)[0]
    ssl_context = _build_stage_ssl_context(tls_cert, tls_key)
    try:
        worker = build_stage_from_manifest(
            manifest_dir,
            stage_id,
            device=stage_device,
            job_id=str(job_id),
            transport=transport,
        )
        endpoint = StageRpcServer(
            worker,
            StageRpcIdentity(
                cluster_id=cluster_id,
                job_id=job_id,
                lease_epoch=lease_epoch,
                worker_incarnation=worker_incarnation,
                peer_worker_incarnation=peer_worker_incarnation,
            ),
            credential=credential,
        )
        scheme = "wss" if ssl_context is not None else "ws"
        click.echo(
            f"Stage {stage_id} RPC server → {scheme}://{host}:{port} "
            f"(device={stage_device})"
        )
        if ssl_context is None:
            click.echo(
                "WARNING: TLS is disabled; use --tls-cert and --tls-key outside "
                "a trusted network.",
                err=True,
            )
        _run_stage_server(endpoint, host, port, ssl_context)
    except (FileNotFoundError, IndexError, KeyError, OSError, RuntimeError, ValueError) as exc:
        raise click.ClickException(f"could not start stage server: {exc}") from exc


def _build_stage_ssl_context(
    tls_cert: Path | None,
    tls_key: Path | None,
) -> ssl.SSLContext | None:
    if (tls_cert is None) != (tls_key is None):
        raise click.ClickException("--tls-cert and --tls-key must be provided together")
    if tls_cert is None:
        return None
    try:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.load_cert_chain(certfile=str(tls_cert), keyfile=str(tls_key))
    except (OSError, ssl.SSLError) as exc:
        raise click.ClickException(f"could not load TLS certificate/key: {exc}") from exc
    return context


def _run_stage_server(endpoint, host: str, port: int, ssl_context: ssl.SSLContext | None) -> None:
    async def serve_forever() -> None:
        websocket_server = await endpoint.serve(host, port, ssl_context=ssl_context)
        try:
            await asyncio.Future()
        finally:
            websocket_server.close()
            await websocket_server.wait_closed()

    try:
        asyncio.run(serve_forever())
    except KeyboardInterrupt:
        click.echo("Stage RPC server stopped.")


# ------------------------------------------------------------------
# remote-serve — HTTP gateway over remote stage endpoints
# ------------------------------------------------------------------

@cli.command("remote-serve")
@click.option(
    "--manifest",
    "manifest_dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Artifact directory containing manifest.json and model metadata.",
)
@click.option(
    "--stage-url",
    "stage_urls",
    multiple=True,
    required=True,
    help="Stage RPC URL in layer order; repeat once per stage.",
)
@click.option(
    "--credential",
    "stage_credentials",
    multiple=True,
    required=True,
    envvar="MESHGPU_STAGE_CREDENTIAL",
    hide_input=True,
    help="Stage credential; one value is reused for all stages, or repeat per stage.",
)
@click.option(
    "--relay-token",
    default=None,
    envvar="MESHGPU_RELAY_TOKEN",
    hide_input=True,
    help="Relay admission token when --stage-url points at WebSocketRelay.",
)
@click.option(
    "--stage-worker-incarnation",
    "stage_worker_incarnations",
    multiple=True,
    required=True,
    type=click.IntRange(0, 0xFFFFFFFF),
    help="Remote worker incarnation; repeat exactly once per stage.",
)
@click.option("--client-incarnation", required=True, type=click.IntRange(0, 0xFFFFFFFF))
@click.option("--cluster-id", required=True, type=click.IntRange(0, 65535))
@click.option("--job-id", required=True, type=click.IntRange(0, 0xFFFFFFFF))
@click.option("--lease-epoch", required=True, type=click.IntRange(0, 0xFFFFFFFF))
@click.option("--host", default="0.0.0.0", show_default=True)
@click.option("--port", default=8090, show_default=True, type=click.IntRange(1, 65535))
@click.option("--kv-slots", default=4096, show_default=True, type=click.IntRange(min=1))
@click.option(
    "--tokenizer",
    "tokenizer_path",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Local tokenizer directory; defaults to the artifact tokenizer.",
)
@click.option(
    "--tls-ca",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="CA bundle for wss:// stage endpoints.",
)
def remote_serve(
    manifest_dir: Path,
    stage_urls: tuple[str, ...],
    stage_credentials: tuple[str, ...],
    relay_token: str | None,
    stage_worker_incarnations: tuple[int, ...],
    client_incarnation: int,
    cluster_id: int,
    job_id: int,
    lease_epoch: int,
    host: str,
    port: int,
    kv_slots: int,
    tokenizer_path: Path | None,
    tls_ca: Path | None,
) -> None:
    """Serve HTTP inference through stage RPC endpoints on other processes/hosts.

    The gateway loads manifest metadata and an optional tokenizer, but never
    loads model weights.  ``--stage-url`` values must be in contiguous layer
    order and match the converted artifact's shard count.
    """
    import uvicorn

    from meshgpu.artifacts.hf_import import validate_manifest_layout
    from meshgpu.artifacts.manifest import ModelManifest
    from meshgpu.backends.portable.rpc import StageRpcIdentity
    from meshgpu.inference.tokenizer import (
        TokenizerError,
        load_tokenizer,
        load_tokenizer_from_artifact,
    )

    try:
        manifest = ModelManifest.load(manifest_dir / "manifest.json")
        n_stages = len(validate_manifest_layout(manifest))
        if len(stage_urls) != n_stages:
            raise ValueError(
                f"artifact has {n_stages} stages, got {len(stage_urls)} --stage-url values"
            )
        credentials = _expand_stage_options(
            stage_credentials,
            n_stages,
            "--credential",
            repeat_single=True,
        )
        worker_incarnations = _expand_stage_options(
            stage_worker_incarnations,
            n_stages,
            "--stage-worker-incarnation",
            repeat_single=False,
        )
        identities = [
            StageRpcIdentity(
                cluster_id=cluster_id,
                job_id=job_id,
                lease_epoch=lease_epoch,
                worker_incarnation=client_incarnation,
                peer_worker_incarnation=worker_incarnations[index],
            )
            for index in range(n_stages)
        ]
        ssl_context = _build_remote_ssl_context(stage_urls, tls_ca)

        tokenizer = None
        if tokenizer_path is not None:
            tokenizer = load_tokenizer(tokenizer_path, local_files_only=True)
        elif manifest.tokenizer_path is not None:
            tokenizer = load_tokenizer_from_artifact(
                manifest_dir,
                manifest.tokenizer_path,
            )
        if tokenizer is not None and tokenizer.vocab_size != manifest.meta.vocab_size:
            raise ValueError(
                f"tokenizer vocab_size={tokenizer.vocab_size} does not match "
                f"model vocab_size={manifest.meta.vocab_size}"
            )
    except (FileNotFoundError, KeyError, OSError, TokenizerError, ValueError) as exc:
        raise click.ClickException(f"could not prepare remote inference: {exc}") from exc

    click.echo(f"Connecting to {n_stages} remote stage(s) …")
    try:
        _run_remote_inference_server(
            stage_urls,
            credentials,
            identities,
            ssl_context=ssl_context,
            relay_token=relay_token,
            vocab_size=manifest.meta.vocab_size,
            max_position_embeddings=manifest.meta.max_position_embeddings,
            tokenizer=tokenizer,
            kv_slots=kv_slots,
            host=host,
            port=port,
            uvicorn_module=uvicorn,
        )
    except (ConnectionError, OSError, RuntimeError, ValueError) as exc:
        raise click.ClickException(f"remote inference gateway failed: {exc}") from exc


def _expand_stage_options(
    values: tuple,
    n_stages: int,
    option_name: str,
    *,
    repeat_single: bool,
) -> list:
    items = list(values)
    if not items:
        raise click.ClickException(f"{option_name} must be provided")
    if repeat_single and len(items) == 1 and n_stages > 1:
        return items * n_stages
    if len(items) != n_stages:
        raise click.ClickException(
            f"need {n_stages} values for {option_name}, got {len(items)}"
        )
    return items


def _build_remote_ssl_context(
    stage_urls: tuple[str, ...],
    tls_ca: Path | None,
) -> ssl.SSLContext | None:
    from urllib.parse import urlparse

    schemes = {urlparse(url).scheme.lower() for url in stage_urls}
    if not schemes or not schemes.issubset({"ws", "wss"}):
        raise click.ClickException("stage URLs must use ws:// or wss://")
    if len(schemes) > 1:
        raise click.ClickException("do not mix ws:// and wss:// stage URLs")
    if schemes == {"ws"}:
        if tls_ca is not None:
            raise click.ClickException("--tls-ca requires wss:// stage URLs")
        click.echo(
            "WARNING: stage RPC TLS is disabled; use wss:// outside a trusted network.",
            err=True,
        )
        return None
    try:
        return ssl.create_default_context(
            cafile=str(tls_ca) if tls_ca is not None else None
        )
    except (OSError, ssl.SSLError) as exc:
        raise click.ClickException(f"could not load stage TLS CA: {exc}") from exc


def _run_remote_inference_server(
    stage_urls: tuple[str, ...],
    credentials: list[str],
    identities: list,
    *,
    ssl_context: ssl.SSLContext | None,
    relay_token: str | None,
    vocab_size: int,
    max_position_embeddings: int,
    tokenizer,
    kv_slots: int,
    host: str,
    port: int,
    uvicorn_module,
) -> None:
    from meshgpu.backends.portable.rpc import connect_stage_clients
    from meshgpu.inference.server import build_inference_app

    async def serve_forever() -> None:
        clients = await connect_stage_clients(
            stage_urls,
            credentials,
            identities,
            ssl_context=ssl_context,
            relay_token=relay_token,
            vocab_size=vocab_size,
            max_position_embeddings=max_position_embeddings,
        )
        try:
            app = build_inference_app(clients, kv_slots=kv_slots, tokenizer=tokenizer)
            config = uvicorn_module.Config(app, host=host, port=port, log_level="info")
            server = uvicorn_module.Server(config)
            click.echo(f"Remote inference gateway → http://{host}:{port}/v1/generate")
            await server.serve()
        finally:
            await asyncio.gather(*(client.close() for client in clients), return_exceptions=True)

    try:
        asyncio.run(serve_forever())
    except KeyboardInterrupt:
        click.echo("Remote inference gateway stopped.")


def _parse_stage_devices(
    devices_spec: str | None,
    fallback: str,
    n_stages: int,
    torch_module,
) -> list:
    """Parse one device per pipeline stage, preserving the old --device path."""
    if n_stages < 1:
        raise click.ClickException("number of stages must be positive")
    names = (
        [part.strip() for part in devices_spec.split(",") if part.strip()]
        if devices_spec is not None
        else [fallback]
    )
    if not names:
        raise click.ClickException("--devices must contain at least one device")
    if len(names) == 1:
        names *= n_stages
    elif len(names) != n_stages:
        raise click.ClickException(
            f"need one device per stage ({n_stages}), got {len(names)} in --devices"
        )
    try:
        parsed = [torch_module.device(name) for name in names]
    except (RuntimeError, TypeError, ValueError) as exc:
        raise click.ClickException(f"invalid device list: {exc}") from exc
    if any(dev.type == "cuda" for dev in parsed) and not torch_module.cuda.is_available():
        raise click.ClickException("CUDA device requested but torch.cuda.is_available() is false")
    cuda_count = torch_module.cuda.device_count() if torch_module.cuda.is_available() else 0
    for name, dev in zip(names, parsed):
        if dev.type == "cuda" and dev.index is not None and dev.index >= cuda_count:
            raise click.ClickException(
                f"device {name!r} is unavailable; torch reports {cuda_count} CUDA device(s)"
            )
    return parsed


# ------------------------------------------------------------------
# fine-tune — real JSONL causal-LM training from a converted artifact
# ------------------------------------------------------------------

@cli.command("fine-tune")
@click.option("--manifest", "manifest_dir", required=True,
              help="Converted MeshGPU artifact directory.")
@click.option("--dataset", "dataset_path", required=True,
              type=click.Path(exists=True, dir_okay=False),
              help="JSONL file of strings or {'text': ...} records.")
@click.option("--out", "output_dir", required=True,
              help="Output directory for checkpoints, export and summary.")
@click.option("--recipe", type=click.Choice(["full", "lora"]), default="lora",
              show_default=True)
@click.option("--devices", default="cpu", show_default=True,
              help="Comma-separated devices, e.g. cuda:0,cuda:1 or cpu.")
@click.option(
    "--transport",
    type=click.Choice(["cpu", "local_cuda"]),
    default="local_cuda",
    show_default=True,
    help="Stage-boundary transport for same-host multi-GPU training.",
)
@click.option(
    "--activation-checkpointing/--no-activation-checkpointing",
    default=False,
    show_default=True,
    help="Recompute decoder blocks during backward to reduce activation memory.",
)
@click.option("--steps", default=100, show_default=True, type=click.IntRange(min=1))
@click.option("--batch-size", default=1, show_default=True, type=click.IntRange(min=1))
@click.option("--sequence-length", default=512, show_default=True,
              type=click.IntRange(min=1))
@click.option("--gradient-accumulation", default=1, show_default=True,
              type=click.IntRange(min=1))
@click.option("--lr", default=2e-4, show_default=True, type=float)
@click.option("--lora-rank", default=16, show_default=True, type=click.IntRange(min=1))
@click.option("--lora-alpha", default=32.0, show_default=True, type=float)
@click.option("--checkpoint-every", default=100, show_default=True,
              type=click.IntRange(min=1))
@click.option(
    "--resume", "resume_checkpoint", default=None,
    help="Committed checkpoint ID in --out/checkpoints to resume from.",
)
@click.option("--tokenizer", "tokenizer_path", default=None,
              help="Tokenizer directory; defaults to the artifact tokenizer.")
def fine_tune(
    manifest_dir: str,
    dataset_path: str,
    output_dir: str,
    recipe: str,
    devices: str,
    transport: str,
    activation_checkpointing: bool,
    steps: int,
    batch_size: int,
    sequence_length: int,
    gradient_accumulation: int,
    lr: float,
    lora_rank: int,
    lora_alpha: float,
    checkpoint_every: int,
    resume_checkpoint: str | None,
    tokenizer_path: str | None,
) -> None:
    """Fine-tune a converted artifact on a local JSONL text dataset."""
    import math
    from pathlib import Path

    import torch

    from meshgpu.artifacts.export import export_checkpoint
    from meshgpu.artifacts.hf_import import load_llama_config
    from meshgpu.artifacts.manifest import ModelManifest
    from meshgpu.backends.native.lora_recipe import LoRAConfig, apply_lora
    from meshgpu.backends.portable.pipeline import build_pipeline_from_manifest
    from meshgpu.backends.portable.trainer import PortableTrainer, TrainerConfig
    from meshgpu.checkpoints.coordinator import CheckpointCoordinator
    from meshgpu.inference.tokenizer import load_tokenizer, load_tokenizer_from_artifact
    from meshgpu.planner.preflight import plan_cuda_training_from_manifest
    from meshgpu.training.data import make_causal_batches

    if not math.isfinite(lr) or lr <= 0:
        raise click.ClickException("--lr must be finite and positive")
    if not math.isfinite(lora_alpha) or lora_alpha <= 0:
        raise click.ClickException("--lora-alpha must be finite and positive")
    manifest_root = Path(manifest_dir)
    manifest = ModelManifest.load(manifest_root / "manifest.json")
    if manifest.meta.adapter == "qwen3_hf_v1":
        from meshgpu.artifacts.qwen_import import load_qwen_config

        cfg = load_qwen_config(manifest_root)
    else:
        cfg = load_llama_config(manifest_root)

    if sequence_length > cfg.max_position_embeddings:
        raise click.ClickException(
            f"--sequence-length ({sequence_length}) exceeds model context "
            f"{cfg.max_position_embeddings}"
        )
    torch_devices = _parse_stage_devices(devices, "cpu", len(manifest.shards), torch)

    planned_layer_ranges = None
    has_cuda = any(device.type == "cuda" for device in torch_devices)
    if has_cuda and not all(device.type == "cuda" for device in torch_devices):
        raise click.ClickException(
            "mixed CPU/CUDA training stages are not supported by the capacity "
            "preflight; use all CPU for correctness tests or one CUDA device per stage"
        )
    if has_cuda:
        try:
            preflight = plan_cuda_training_from_manifest(
                manifest_root,
                torch_devices,
                batch_size=batch_size,
                sequence_length=sequence_length,
                gradient_accumulation_steps=gradient_accumulation,
                activation_checkpointing=activation_checkpointing,
                recipe=recipe,
                lora_rank=lora_rank,
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise click.ClickException(
                f"CUDA training preflight failed before loading weights: {exc}"
            ) from exc
        click.echo("CUDA training preflight:")
        click.echo(preflight.summary())
        if not preflight.feasible:
            raise click.ClickException(
                "training rejected before loading weights: " + preflight.reason
            )
        planned_layer_ranges = [
            (assignment.layer_start, assignment.layer_end)
            for assignment in preflight.assignments
        ]

    workers = build_pipeline_from_manifest(
        manifest_root,
        devices=torch_devices,
        transport=transport,
        activation_checkpointing=activation_checkpointing,
        layer_ranges=planned_layer_ranges,
    )
    lora_cfg: LoRAConfig | None = None
    if recipe == "lora":
        lora_cfg = LoRAConfig(rank=lora_rank, alpha=lora_alpha)
        for worker in workers:
            apply_lora(worker._model, lora_cfg)

    optimizers: list[torch.optim.Optimizer] = []
    for worker in workers:
        trainable = [
            parameter
            for parameter in worker._model.parameters()
            if parameter.requires_grad
        ]
        if not trainable:
            raise click.ClickException(f"stage {worker._ctx.stage_id} has no trainable parameters")
        optimizers.append(torch.optim.AdamW(trainable, lr=lr))

    if tokenizer_path is not None:
        tokenizer_dir = Path(tokenizer_path)
        if not tokenizer_dir.is_dir():
            raise click.ClickException(
                f"--tokenizer must point to a local tokenizer directory: {tokenizer_dir}"
            )
        tokenizer = load_tokenizer(tokenizer_dir, local_files_only=True)
    elif manifest.tokenizer_path is not None:
        tokenizer = load_tokenizer_from_artifact(manifest_root, manifest.tokenizer_path)
    else:
        raise click.ClickException(
            "artifact has no tokenizer; reconvert with tokenizer or pass --tokenizer"
        )
    if tokenizer.vocab_size != cfg.vocab_size:
        raise click.ClickException(
            f"tokenizer vocab_size={tokenizer.vocab_size} does not match model "
            f"vocab_size={cfg.vocab_size}"
        )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    coordinator = CheckpointCoordinator(output / "checkpoints")
    trainer = PortableTrainer(
        workers,
        optimizers,
        checkpoint=coordinator,
        cfg=TrainerConfig(
            max_steps=steps,
            gradient_accumulation_steps=gradient_accumulation,
            checkpoint_every_steps=checkpoint_every,
            log_every_steps=1,
        ),
        job_id=f"fine_tune:{manifest.meta.model_id}",
    )
    if resume_checkpoint is not None:
        try:
            trainer.resume(resume_checkpoint)
        except (FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
            raise click.ClickException(
                f"could not resume checkpoint {resume_checkpoint!r}: {exc}"
            ) from exc
        click.echo(
            f"Resumed checkpoint {resume_checkpoint} at global step {trainer.global_step}; "
            f"continuing after {trainer.data_cursor} packed data batch(es)."
        )
        if trainer.global_step >= steps:
            raise click.ClickException(
                f"--steps ({steps}) must be greater than resumed global step "
                f"{trainer.global_step}"
            )
    batches = make_causal_batches(
        dataset_path,
        tokenizer,
        sequence_length=sequence_length,
        batch_size=batch_size,
        repeat=True,
        drop_last=True,
        skip_batches=trainer.data_cursor,
    )

    last_checkpoint_id: str | None = None
    last_checkpoint_step: int | None = None
    last_step = 0
    for result in trainer.train(batches):
        last_step = result.step
        if result.checkpoint_id is not None:
            last_checkpoint_id = result.checkpoint_id
            last_checkpoint_step = result.step
        click.echo(
            f"step={result.step} loss={result.loss:.4f} "
            f"grad_norm={result.grad_norm:.3f} tok/s={result.tokens_per_second:.1f}"
        )

    if last_step == 0:
        raise click.ClickException("training produced no optimizer step")
    if last_checkpoint_id is None or last_checkpoint_step != last_step:
        last_checkpoint_id = trainer.save_checkpoint()

    export_dir = output / "inference"
    export_checkpoint(
        coordinator,
        last_checkpoint_id,
        export_dir,
        cfg,
        num_stages=len(workers),
        compute_dtype=manifest.meta.compute_dtype,
        model_id=manifest.meta.model_id,
        tokenizer_source=(
            manifest_root / manifest.tokenizer_path
            if manifest.tokenizer_path is not None and tokenizer_path is None
            else tokenizer_path
        ),
        lora_config=lora_cfg,
        attn_implementation=(
            str(manifest.meta.extra["attn_implementation"])
            if "attn_implementation" in manifest.meta.extra
            else None
        ),
    )
    summary = {
        "recipe": recipe,
        "global_step": trainer.global_step,
        "checkpoint_id": last_checkpoint_id,
        "checkpoint_dir": str(output / "checkpoints"),
        "inference_manifest": str(export_dir / "manifest.json"),
        "model_id": manifest.meta.model_id,
        "tokenizer_vocab_size": tokenizer.vocab_size,
    }
    (output / "training_summary.json").write_text(json.dumps(summary, indent=2))
    click.echo(f"Done. Checkpoint: {last_checkpoint_id}")
    click.echo(f"Inference artifact: {export_dir / 'manifest.json'}")


# ------------------------------------------------------------------
# benchmark
# ------------------------------------------------------------------

@cli.command()
@click.option("--model", type=click.Choice(["tiny", "default"]), default="tiny",
              show_default=True)
@click.option("--stages", default=2, show_default=True, type=click.IntRange(min=1))
@click.option("--device", default="cpu", show_default=True)
@click.option("--quick/--full", default=False, show_default=True)
def benchmark(model: str, stages: int, device: str, quick: bool) -> None:
    """Run the portable pipeline benchmark suite."""
    import torch

    from meshgpu.benchmarks.throughput import run_all

    # Validate the device before starting a potentially long benchmark.  The
    # benchmark module still receives one device string because it constructs
    # a homogeneous stage list for this CLI path.
    stage_devices = _parse_stage_devices(None, device, stages, torch)
    report = run_all(
        cfg=None if model == "default" else _tiny_benchmark_config(),
        num_stages=stages,
        device=str(stage_devices[0]),
        quick=quick,
    )
    click.echo(report.summary())


def _tiny_benchmark_config():
    from meshgpu.models.llama_dense import LlamaConfig

    return LlamaConfig(
        vocab_size=1024,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        max_position_embeddings=512,
    )


# ------------------------------------------------------------------
# submit
# ------------------------------------------------------------------

@cli.command()
@click.argument("job_yaml")
@click.option("--controller", "controller_url", required=True, envvar="MESHGPU_CONTROLLER")
def submit(job_yaml: str, controller_url: str) -> None:
    """Submit a job from a YAML spec."""
    try:
        import yaml
    except ImportError:
        click.echo("pyyaml not installed", err=True)
        sys.exit(1)
    import urllib.request

    with open(job_yaml) as f:
        spec = yaml.safe_load(f)

    body = json.dumps(spec).encode()
    req = urllib.request.Request(
        f"{controller_url.rstrip('/')}/v1/jobs",
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read())
    click.echo(json.dumps(data, indent=2))


if __name__ == "__main__":
    cli()
