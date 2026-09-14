"""
Provider eligibility matrix.
Determines which modes (client / single_runtime_job / distributed_worker)
a given provider + environment combination actually supports.

Key rules from plan.md §3:
  - Colab managed (free, no compute units): ONLY client or single notebook job.
    Running distributed workers is listed in Colab FAQ as disallowed.
  - Colab paid / local runtime: conditional; must smoke-test per-workflow.
  - Kaggle: session-level Internet toggle; check per-account and per-competition.
  - Self-managed Linux NVIDIA: full support after preflight.
  - WSL2, macOS, AMD: staged / gated.
"""
from __future__ import annotations

import importlib.util
import logging
import os
from dataclasses import dataclass
from enum import Enum

log = logging.getLogger(__name__)


class ProviderKind(str, Enum):
    SELF_MANAGED = "self_managed"
    COLAB_MANAGED_FREE = "colab_managed_free"
    COLAB_MANAGED_PAID = "colab_managed_paid"
    COLAB_LOCAL_RUNTIME = "colab_local_runtime"
    KAGGLE = "kaggle"
    CLOUD_VM = "cloud_vm"
    UNKNOWN = "unknown"


class SupportLevel(str, Enum):
    SUPPORTED = "supported"
    CONDITIONAL = "conditional"   # requires extra smoke-test
    EXPERIMENTAL = "experimental" # may work but not covered by QA
    UNSUPPORTED = "unsupported"
    CLIENT_ONLY = "client_only"   # can connect as SDK client, not as worker


@dataclass
class EligibilityResult:
    provider: ProviderKind
    distributed_worker: SupportLevel
    single_runtime_job: SupportLevel
    client: SupportLevel
    reason: str
    outbound_relay_required: bool
    internet_available: bool | None  # None = unknown


# ---------------------------------------------------------------------------
# Matrix
# ---------------------------------------------------------------------------

_MATRIX: dict[ProviderKind, EligibilityResult] = {
    ProviderKind.SELF_MANAGED: EligibilityResult(
        provider=ProviderKind.SELF_MANAGED,
        distributed_worker=SupportLevel.SUPPORTED,
        single_runtime_job=SupportLevel.SUPPORTED,
        client=SupportLevel.SUPPORTED,
        reason="Full support after preflight passes",
        outbound_relay_required=False,
        internet_available=None,
    ),
    ProviderKind.CLOUD_VM: EligibilityResult(
        provider=ProviderKind.CLOUD_VM,
        distributed_worker=SupportLevel.SUPPORTED,
        single_runtime_job=SupportLevel.SUPPORTED,
        client=SupportLevel.SUPPORTED,
        reason="Treated as self_managed; verify GPU driver and network",
        outbound_relay_required=False,
        internet_available=True,
    ),
    ProviderKind.COLAB_MANAGED_FREE: EligibilityResult(
        provider=ProviderKind.COLAB_MANAGED_FREE,
        distributed_worker=SupportLevel.UNSUPPORTED,
        single_runtime_job=SupportLevel.CLIENT_ONLY,
        client=SupportLevel.SUPPORTED,
        reason=(
            "Colab FAQ lists running distributed computing workers as disallowed "
            "on free managed runtimes. Only client-mode is safe."
        ),
        outbound_relay_required=True,
        internet_available=True,
    ),
    ProviderKind.COLAB_MANAGED_PAID: EligibilityResult(
        provider=ProviderKind.COLAB_MANAGED_PAID,
        distributed_worker=SupportLevel.CONDITIONAL,
        single_runtime_job=SupportLevel.CONDITIONAL,
        client=SupportLevel.SUPPORTED,
        reason=(
            "Paid Colab still subject to provider ToS and remote-proxy restrictions. "
            "Run eligibility smoke-test before adding to pool."
        ),
        outbound_relay_required=True,
        internet_available=True,
    ),
    ProviderKind.COLAB_LOCAL_RUNTIME: EligibilityResult(
        provider=ProviderKind.COLAB_LOCAL_RUNTIME,
        distributed_worker=SupportLevel.SUPPORTED,
        single_runtime_job=SupportLevel.SUPPORTED,
        client=SupportLevel.SUPPORTED,
        reason="Local runtime is user-owned hardware; Colab UI is just the frontend",
        outbound_relay_required=False,
        internet_available=None,
    ),
    ProviderKind.KAGGLE: EligibilityResult(
        provider=ProviderKind.KAGGLE,
        distributed_worker=SupportLevel.EXPERIMENTAL,
        single_runtime_job=SupportLevel.EXPERIMENTAL,
        client=SupportLevel.SUPPORTED,
        reason=(
            "Kaggle internet toggle is session-level; some competitions require it off. "
            "Verify per-account quota, session duration and network access."
        ),
        outbound_relay_required=True,
        internet_available=None,  # depends on session config
    ),
    ProviderKind.UNKNOWN: EligibilityResult(
        provider=ProviderKind.UNKNOWN,
        distributed_worker=SupportLevel.UNSUPPORTED,
        single_runtime_job=SupportLevel.UNSUPPORTED,
        client=SupportLevel.UNSUPPORTED,
        reason="Unknown provider — run preflight to identify environment",
        outbound_relay_required=True,
        internet_available=None,
    ),
}


def detect_provider() -> ProviderKind:
    """Heuristic detection of the current execution environment."""
    explicit = os.environ.get("MESHGPU_PROVIDER")
    if explicit:
        try:
            return ProviderKind(explicit)
        except ValueError:
            log.warning("unknown MESHGPU_PROVIDER=%r; falling back to detection", explicit)

    if "COLAB_BACKEND_URL" in os.environ or "COLAB_GPU" in os.environ:
        # Do not pin detection to the Python minor version used by one Colab
        # image.  ``find_spec`` also avoids importing google.colab (which can
        # have side effects in a notebook).  The environment marker is kept
        # as a second signal for images that do not expose the package spec.
        try:
            colab_spec = importlib.util.find_spec("google.colab")
        except (ImportError, ModuleNotFoundError, AttributeError):
            colab_spec = None
        if colab_spec is not None or "COLAB_RELEASE_TAG" in os.environ:
            # Managed runtime
            has_compute = os.environ.get("COLAB_COMPUTE_UNITS_POSITIVE", "0") == "1"
            return (
                ProviderKind.COLAB_MANAGED_PAID
                if has_compute
                else ProviderKind.COLAB_MANAGED_FREE
            )
        return ProviderKind.COLAB_LOCAL_RUNTIME
    if "KAGGLE_KERNEL_RUN_TYPE" in os.environ or "KAGGLE_DATA_PROXY_PROJECT" in os.environ:
        return ProviderKind.KAGGLE
    return ProviderKind.SELF_MANAGED


def check_eligibility(
    provider: ProviderKind | None = None,
) -> EligibilityResult:
    """Return eligibility for current or given provider."""
    kind = provider or detect_provider()
    result = _MATRIX.get(kind, _MATRIX[ProviderKind.UNKNOWN])
    if result.distributed_worker in (SupportLevel.UNSUPPORTED, SupportLevel.CLIENT_ONLY):
        log.warning(
            "provider %s: distributed_worker=%s — %s",
            kind.value, result.distributed_worker.value, result.reason,
        )
    return result


def assert_eligible_for_worker(provider: ProviderKind | None = None) -> None:
    """Raise if this provider cannot be a distributed worker."""
    result = check_eligibility(provider)
    if result.distributed_worker in (SupportLevel.UNSUPPORTED, SupportLevel.CLIENT_ONLY):
        raise RuntimeError(
            f"Provider {result.provider.value} cannot act as a distributed worker: "
            f"{result.reason}"
        )
    if result.distributed_worker in (SupportLevel.CONDITIONAL, SupportLevel.EXPERIMENTAL):
        log.warning(
            "provider %s distributed_worker support is %s — run smoke-test before adding to pool",
            result.provider.value, result.distributed_worker.value,
        )
