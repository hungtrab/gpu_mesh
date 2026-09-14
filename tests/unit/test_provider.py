"""Tests for provider eligibility matrix."""
import pytest

from meshgpu.agent.probe import build_capability_report
from meshgpu.agent.provider import (
    ProviderKind,
    SupportLevel,
    assert_eligible_for_worker,
    check_eligibility,
    detect_provider,
)
from meshgpu.protocol.messages import WorkerMode


class TestDetectProvider:
    def test_default_self_managed(self, monkeypatch):
        for key in ("COLAB_BACKEND_URL", "COLAB_GPU", "KAGGLE_KERNEL_RUN_TYPE",
                    "KAGGLE_DATA_PROXY_PROJECT", "MESHGPU_PROVIDER"):
            monkeypatch.delenv(key, raising=False)
        assert detect_provider() == ProviderKind.SELF_MANAGED

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("MESHGPU_PROVIDER", "cloud_vm")
        assert detect_provider() == ProviderKind.CLOUD_VM

    def test_env_override_invalid_falls_back(self, monkeypatch):
        monkeypatch.setenv("MESHGPU_PROVIDER", "totally_invalid_provider")
        # Should fall back to self_managed (not raise)
        result = detect_provider()
        assert result == ProviderKind.SELF_MANAGED

    def test_kaggle_detection(self, monkeypatch):
        monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Interactive")
        monkeypatch.delenv("COLAB_BACKEND_URL", raising=False)
        monkeypatch.delenv("COLAB_GPU", raising=False)
        assert detect_provider() == ProviderKind.KAGGLE

    def test_colab_detection_does_not_depend_on_python_minor_version(self, monkeypatch):
        monkeypatch.setenv("COLAB_GPU", "T4")
        monkeypatch.setenv("COLAB_RELEASE_TAG", "release-2026")
        monkeypatch.delenv("COLAB_COMPUTE_UNITS_POSITIVE", raising=False)
        assert detect_provider() == ProviderKind.COLAB_MANAGED_FREE


class TestEligibilityMatrix:
    def test_self_managed_fully_supported(self):
        r = check_eligibility(ProviderKind.SELF_MANAGED)
        assert r.distributed_worker == SupportLevel.SUPPORTED
        assert r.client == SupportLevel.SUPPORTED
        assert r.outbound_relay_required is False

    def test_colab_free_worker_unsupported(self):
        r = check_eligibility(ProviderKind.COLAB_MANAGED_FREE)
        assert r.distributed_worker == SupportLevel.UNSUPPORTED
        assert r.client == SupportLevel.SUPPORTED
        assert r.outbound_relay_required is True

    def test_colab_paid_worker_conditional(self):
        r = check_eligibility(ProviderKind.COLAB_MANAGED_PAID)
        assert r.distributed_worker == SupportLevel.CONDITIONAL

    def test_colab_local_runtime_fully_supported(self):
        r = check_eligibility(ProviderKind.COLAB_LOCAL_RUNTIME)
        assert r.distributed_worker == SupportLevel.SUPPORTED
        assert r.outbound_relay_required is False

    def test_kaggle_experimental(self):
        r = check_eligibility(ProviderKind.KAGGLE)
        assert r.distributed_worker == SupportLevel.EXPERIMENTAL

    def test_cloud_vm_supported(self):
        r = check_eligibility(ProviderKind.CLOUD_VM)
        assert r.distributed_worker == SupportLevel.SUPPORTED

    def test_unknown_unsupported(self):
        r = check_eligibility(ProviderKind.UNKNOWN)
        assert r.distributed_worker == SupportLevel.UNSUPPORTED


class TestAssertEligibleForWorker:
    def test_self_managed_passes(self):
        assert_eligible_for_worker(ProviderKind.SELF_MANAGED)  # no raise

    def test_cloud_vm_passes(self):
        assert_eligible_for_worker(ProviderKind.CLOUD_VM)

    def test_colab_local_passes(self):
        assert_eligible_for_worker(ProviderKind.COLAB_LOCAL_RUNTIME)

    def test_colab_free_raises(self):
        with pytest.raises(RuntimeError, match="cannot act as a distributed worker"):
            assert_eligible_for_worker(ProviderKind.COLAB_MANAGED_FREE)

    def test_unknown_raises(self):
        with pytest.raises(RuntimeError):
            assert_eligible_for_worker(ProviderKind.UNKNOWN)

    def test_kaggle_warns_but_does_not_raise(self, caplog):
        import logging
        with caplog.at_level(logging.WARNING, logger="meshgpu.agent.provider"):
            assert_eligible_for_worker(ProviderKind.KAGGLE)
        assert any("experimental" in m.lower() for m in caplog.messages)

    def test_colab_paid_warns_but_does_not_raise(self, caplog):
        import logging
        with caplog.at_level(logging.WARNING, logger="meshgpu.agent.provider"):
            assert_eligible_for_worker(ProviderKind.COLAB_MANAGED_PAID)
        assert any("conditional" in m.lower() for m in caplog.messages)


def test_capability_report_does_not_advertise_free_colab_as_worker(monkeypatch):
    monkeypatch.setattr("meshgpu.agent.probe.probe_gpus", lambda: [])
    report = build_capability_report(provider=ProviderKind.COLAB_MANAGED_FREE.value)
    assert WorkerMode.CLIENT in report.supported_modes
    assert WorkerMode.SINGLE_RUNTIME_JOB in report.supported_modes
    assert WorkerMode.DISTRIBUTED_WORKER not in report.supported_modes


def test_capability_report_advertises_worker_for_self_managed(monkeypatch):
    monkeypatch.setattr("meshgpu.agent.probe.probe_gpus", lambda: [])
    report = build_capability_report(provider=ProviderKind.SELF_MANAGED.value)
    assert WorkerMode.DISTRIBUTED_WORKER in report.supported_modes
