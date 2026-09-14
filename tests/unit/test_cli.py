"""CLI-level smoke tests for the supported local workflows."""

import json

import pytest
from click.exceptions import ClickException
from click.testing import CliRunner

from meshgpu.cli.main import (
    _build_remote_ssl_context,
    _expand_stage_options,
    cli,
)


def _job_yaml() -> str:
    return """
task: inference
model:
  num_layers: 4
  hidden_size: 64
  intermediate_size: 128
  num_attention_heads: 4
  num_kv_heads: 2
  head_dim: 16
  vocab_size: 256
  dtype_bytes: 2
placement:
  workers:
    - worker_id: gpu-a
      total_vram_gib: 16
    - worker_id: gpu-b
      total_vram_gib: 16
inference:
  max_prompt_tokens: 64
  max_new_tokens: 8
"""


def test_plan_command_returns_json_report(tmp_path):
    job = tmp_path / "job.yaml"
    job.write_text(_job_yaml())

    result = CliRunner().invoke(cli, ["plan", str(job)])

    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["feasible"] is True
    assert len(report["assignments"]) == 2
    assert report["summary"].startswith("feasible=True")


def test_plan_command_reports_yaml_errors(tmp_path):
    job = tmp_path / "broken.yaml"
    job.write_text("placement: [\n")

    result = CliRunner().invoke(cli, ["plan", str(job)])

    assert result.exit_code != 0
    assert "could not build placement plan" in result.output


def test_benchmark_rejects_invalid_device_before_work(tmp_path):
    result = CliRunner().invoke(
        cli,
        ["benchmark", "--stages", "2", "--device", "not-a-device", "--quick"],
    )

    assert result.exit_code != 0
    assert "invalid device list" in result.output


def test_python_module_entrypoint_is_registered():
    result = CliRunner().invoke(cli, ["--help"])

    assert result.exit_code == 0
    assert "fine-tune" in result.output
    assert "benchmark" in result.output


def test_stage_server_help_exposes_single_shard_entrypoint():
    result = CliRunner().invoke(cli, ["stage-server", "--help"])

    assert result.exit_code == 0, result.output
    assert "loads only the selected shard" in result.output
    assert "--credential" in result.output
    assert "--worker-incarnation" in result.output


def test_stage_worker_help_exposes_outbound_relay_entrypoint():
    result = CliRunner().invoke(cli, ["stage-worker", "--help"])

    assert result.exit_code == 0, result.output
    assert "connect outbound" in result.output
    assert "--relay-url" in result.output
    assert "--relay-token" in result.output


def test_relay_server_help_exposes_tls_and_token_options():
    result = CliRunner().invoke(cli, ["relay-server", "--help"])

    assert result.exit_code == 0, result.output
    assert "outbound worker/gateway" in result.output
    assert "--relay-token" in result.output
    assert "--tls-cert" in result.output


def test_remote_serve_help_exposes_metadata_only_gateway():
    result = CliRunner().invoke(cli, ["remote-serve", "--help"])

    assert result.exit_code == 0, result.output
    assert "never" in result.output
    assert "model weights" in result.output
    assert "--stage-url" in result.output
    assert "--relay-token" in result.output


def test_expand_stage_options_only_repeats_explicitly_allowed_values():
    assert _expand_stage_options(("shared",), 2, "--credential", repeat_single=True) == [
        "shared",
        "shared",
    ]
    with pytest.raises(ClickException, match="need 2 values"):
        _expand_stage_options((101,), 2, "--stage-worker-incarnation", repeat_single=False)


def test_remote_ssl_context_rejects_mixed_or_inconsistent_options(tmp_path):
    ca = tmp_path / "ca.pem"
    ca.write_text("not a certificate")

    with pytest.raises(ClickException, match="do not mix"):
        _build_remote_ssl_context(("ws://a", "wss://b"), None)
    with pytest.raises(ClickException, match="requires wss"):
        _build_remote_ssl_context(("ws://a",), ca)
    with pytest.raises(ClickException, match="ws:// or wss://"):
        _build_remote_ssl_context(("http://a",), None)
