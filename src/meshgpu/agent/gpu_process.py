"""Small, deliberately boring GPU subprocess entry point.

The supervisor owns lifecycle; model loading and stage RPC are separate concerns.
Keeping this process alive on the selected device gives the agent a real resource
boundary without pretending that a heartbeat connection is a model executor.
"""
from __future__ import annotations

import logging
import os
import signal
from multiprocessing.connection import Connection

log = logging.getLogger(__name__)


def run_gpu_process(device_index: int, commands: Connection) -> None:
    """Own ``device_index`` until the parent sends ``stop`` or closes the pipe."""
    stopping = False

    def _request_stop(_signum: int, _frame) -> None:
        nonlocal stopping
        stopping = True

    for signum in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(signum, _request_stop)
        except (OSError, ValueError):
            # Signal registration can be unavailable in an embedded runtime;
            # the command pipe remains the normal shutdown path.
            pass

    try:
        import torch

        if device_index < 0:
            raise ValueError(f"device_index must be non-negative, got {device_index}")
        if torch.cuda.is_available():
            if device_index >= torch.cuda.device_count():
                raise ValueError(
                    f"device_index {device_index} is unavailable; "
                    f"torch reports {torch.cuda.device_count()} CUDA device(s)"
                )
            torch.cuda.set_device(device_index)
        elif device_index != 0:
            # CPU-only development can use logical device 0.  Rejecting other
            # indices prevents a silently misconfigured agent.
            raise RuntimeError(
                "CUDA is unavailable; CPU-only GPU subprocesses must use device_index=0"
            )

        commands.send({
            "event": "ready",
            "pid": os.getpid(),
            "device_index": device_index,
            "cuda": bool(torch.cuda.is_available()),
        })

        while not stopping:
            if commands.poll(0.5):
                try:
                    command = commands.recv()
                except (EOFError, OSError):
                    break
                if command == "stop":
                    break
                log.warning("ignoring unknown GPU process command: %r", command)
    except Exception as exc:
        log.exception("GPU process failed during startup: %s", exc)
        try:
            commands.send({"event": "error", "message": str(exc)})
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        try:
            commands.close()
        except OSError:
            pass
