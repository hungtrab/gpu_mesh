"""Fresh GPU preflight for the second Kaggle account."""
from __future__ import annotations

import os
import socket

import torch

print(f"torch={torch.__version__} cuda={torch.version.cuda}", flush=True)
print(
    f"cuda_available={torch.cuda.is_available()} "
    f"device_count={torch.cuda.device_count()} "
    f"visible={os.environ.get('CUDA_VISIBLE_DEVICES')!r}",
    flush=True,
)
for index in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(index)
    print(
        f"device[{index}]={props.name} "
        f"memory={props.total_memory} capability={props.major}.{props.minor}",
        flush=True,
    )
if torch.cuda.is_available():
    value = torch.randn((2048, 2048), device="cuda")
    result = (value @ value.T).mean()
    torch.cuda.synchronize()
    print(f"cuda_smoke={float(result):.6f}", flush=True)
print(f"hostname={socket.gethostname()}", flush=True)
