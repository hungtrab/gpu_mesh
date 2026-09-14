"""MeshGPU WAN smoke worker: Kaggle session B owns stage 1."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

MODEL_ID = "Qwen/Qwen3-0.6B"
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
JOB_ID = 4242
CLUSTER_ID = 7
LEASE_EPOCH = 1
WORKER_INCARNATION = 1002
PEER_INCARNATION = 3001
ARTIFACT_ROOT = Path("/kaggle/working/qwen3-0.6b-2s")
SOURCE_ROOT = Path("/kaggle/working/meshgpu-source")
SOURCE_ARCHIVE = Path("/kaggle/working/meshgpu-source.tar.gz")


def _secret(name: str) -> str:
    environment_name = f"MESHGPU_{name}"
    value = os.environ.get(environment_name)
    if value:
        return value
    try:
        from kaggle_secrets import UserSecretsClient

        value = UserSecretsClient().get_secret(environment_name)
    except Exception as exc:
        raise RuntimeError(
            f"missing Kaggle Secret {environment_name!r}; "
            "configure the relay and stage credentials before starting"
        ) from exc
    if not value:
        raise RuntimeError(f"Kaggle Secret {environment_name!r} is empty")
    return value


def _install_meshgpu() -> None:
    if not (SOURCE_ROOT / "pyproject.toml").exists():
        SOURCE_ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
        bundled_roots = sorted(
            {
                path.parent
                for path in Path("/kaggle/input").glob("**/pyproject.toml")
                if (path.parent / "src" / "meshgpu").is_dir()
            }
        )
        if bundled_roots:
            shutil.copytree(bundled_roots[0], SOURCE_ROOT, symlinks=False)
        else:
            bundled = sorted(
                Path("/kaggle/input").glob("**/meshgpu-kaggle-smoke.tar.gz")
            )
            if bundled:
                shutil.copyfile(bundled[0], SOURCE_ARCHIVE)
            else:
                urllib.request.urlretrieve(_secret("SOURCE_URL"), SOURCE_ARCHIVE)
            SOURCE_ROOT.mkdir(parents=True, exist_ok=True)
            with tarfile.open(SOURCE_ARCHIVE, "r:gz") as archive:
                members = archive.getmembers()
                root = SOURCE_ROOT.resolve()
                for member in members:
                    if member.issym() or member.islnk():
                        raise RuntimeError("source archive must not contain symlinks")
                target = (SOURCE_ROOT / member.name).resolve()
                target.relative_to(root)
                archive.extractall(SOURCE_ROOT)
    source_path = str(SOURCE_ROOT / "src")
    existing_pythonpath = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = (
        source_path
        if not existing_pythonpath
        else os.pathsep.join((source_path, existing_pythonpath))
    )
    transformers_probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import transformers; from transformers import Qwen3Config",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if transformers_probe.returncode != 0:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-input",
                "--only-binary=:all:",
                "-q",
                "transformers>=4.56,<5",
            ],
            check=True,
        )


def _prepare_artifact() -> None:
    if (ARTIFACT_ROOT / "manifest.json").exists():
        return
    if ARTIFACT_ROOT.exists():
        shutil.rmtree(ARTIFACT_ROOT)
    bundled = sorted(
        path.parent
        for path in Path("/kaggle/input").glob("**/manifest.json")
        if (path.parent / "config_36a39304.json").is_file()
        and (path.parent / "stage_00_36a39304.pt").is_file()
        and (path.parent / "stage_01_36a39304.pt").is_file()
    )
    if bundled:
        shutil.copytree(bundled[0], ARTIFACT_ROOT, symlinks=False)
        print(f"meshgpu artifact: using bundled dataset {bundled[0]}", flush=True)
        return
    subprocess.run(
        [
            sys.executable,
            "-m",
            "meshgpu.cli.main",
            "convert",
            MODEL_ID,
            "--revision",
            MODEL_REVISION,
            "--stages",
            "2",
            "--dtype",
            "float16",
            "--attn-implementation",
            "sdpa",
            "--no-tokenizer",
            "--out",
            str(ARTIFACT_ROOT),
        ],
        check=True,
    )


def main() -> None:
    os.environ["MESHGPU_PROVIDER"] = "kaggle"
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Kaggle GPU preflight failed: no CUDA device was attached; "
            "enable a GPU runtime for this account before starting MeshGPU"
        )
    print(
        f"meshgpu provider=kaggle gpu_count={torch.cuda.device_count()} "
        f"cuda={torch.version.cuda}",
        flush=True,
    )
    print("meshgpu bootstrap: preparing source and HF runtime", flush=True)
    _install_meshgpu()
    print("meshgpu bootstrap: runtime ready", flush=True)
    _prepare_artifact()
    relay_url = _secret("RELAY_URL")
    environment = os.environ.copy()
    environment["MESHGPU_RELAY_TOKEN"] = _secret("RELAY_TOKEN")
    environment["MESHGPU_STAGE_CREDENTIAL"] = _secret("STAGE_CREDENTIAL")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "meshgpu.cli.main",
            "stage-worker",
            "--manifest",
            str(ARTIFACT_ROOT),
            "--stage",
            "1",
            "--relay-url",
            relay_url,
            "--device",
            "cuda:0",
            "--cluster-id",
            str(CLUSTER_ID),
            "--job-id",
            str(JOB_ID),
            "--lease-epoch",
            str(LEASE_EPOCH),
            "--worker-incarnation",
            str(WORKER_INCARNATION),
            "--peer-worker-incarnation",
            str(PEER_INCARNATION),
            "--max-reconnect-attempts",
            "10",
        ],
        check=True,
        env=environment,
    )


if __name__ == "__main__":
    main()
