# MeshGPU

MeshGPU là runtime thử nghiệm để chia một dense decoder model thành các stage liên tiếp.
Mỗi stage giữ embedding/layer/head được giao cho mình; hidden state là dữ liệu đi qua
ranh giới stage, còn KV cache nằm tại stage sở hữu các layer đó.

Hướng dẫn thao tác đầy đủ cho người dùng và agent nằm trong
[`USAGE.md`](USAGE.md), gồm inference, fine-tune, planner, stage RPC, hai Kaggle
session qua relay, secret contract, recovery và acceptance checklist.

## Trạng thái hiện tại

Đường đã được kiểm chứng tự động:

- import model Llama từ Hugging Face và chia thành các shard theo layer;
- chạy inference prefill/decode, greedy hoặc top-p, streaming SSE;
- chạy LoRA hoặc full fine-tuning bằng portable pipeline;
- checkpoint có hash và commit pointer nguyên tử, resume cùng topology;
- reshard full-fine-tuning checkpoint qua topology khác; LoRA checkpoint sẽ fail
  rõ ràng vì adapter configuration chưa được lưu đủ để remap an toàn;
- export checkpoint thành artifact inference và load lại;
- planner ước lượng VRAM cho inference/training.

Đường mặc định hiện là portable pipeline trong một Python process, phù hợp để dùng
nhiều GPU trên một máy và làm correctness reference. Controller, agent, relay và
transport là nền tảng cho multi-machine; direct cross-machine stage execution chưa
được quảng bá là production-ready. QUIC/RDMA vẫn là integration point thử nghiệm.

Hai chế độ workload được tách riêng: `serve` chạy inference với KV cache riêng cho
từng request; `fine-tune` chạy full hoặc LoRA training với optimizer/checkpoint riêng.
Thêm GPU trước hết giúp chứa resident weights và state của model lớn hơn, không tự
đảm bảo latency hay throughput tăng tuyến tính.

Stage RPC qua WebSocket/TLS có API async để nối các stage ở process hoặc host khác.
Nó dùng frame chunking, checksum, lease/incarnation validation và credit-based
backpressure. Inference và explicit LoRA TTT đều dùng cùng endpoint; backward không
đi xuyên autograd graph mà dùng activation/gradient protocol tường minh:

```python
from meshgpu.backends.portable.pipeline import pipeline_prefill_async
from meshgpu.backends.portable.rpc import StageRpcClient, StageRpcIdentity

stage = await StageRpcClient.connect(
    "wss://stage.example/v1/stage",
    credential,
    StageRpcIdentity(
        cluster_id=1, job_id=42, lease_epoch=3,
        worker_incarnation=101, peer_worker_incarnation=202,
    ),
)
logits = await pipeline_prefill_async([stage], prompt_ids)
await stage.close()
```

Remote TTT giữ loss/logits và autograd graph tại stage cuối, chỉ gửi hidden state
tiến và boundary gradient lùi. `remote-ttt` mặc định dùng NVARC-style
`rank=256`, `alpha=32`, rsLoRA, bảy projection attention/MLP và lưu đầy đủ
`embed_tokens`/`lm_head`; đây là đường experimental và cần một client độc quyền
cho mỗi stage:

```bash
meshgpu remote-ttt \
  --manifest ./artifacts/qwen3-4b-2s \
  --stage-url "$STAGE0_URL" --stage-url "$STAGE1_URL" \
  --credential "$STAGE0_SECRET" --credential "$STAGE1_SECRET" \
  --relay-token "$RELAY_TOKEN" \
  --stage-worker-incarnation 1001 --stage-worker-incarnation 1002 \
  --client-incarnation 3001 --cluster-id 7 --job-id 4242 --lease-epoch 1 \
  --batch ./data/one_task.json --steps 1
```

File batch là JSON thuần với `input_ids` và tùy chọn `labels`; gateway không đọc
weights. Với ứng dụng nhiều task, dùng `RemoteTaskTTTSession` và gọi
`reset_task()` giữa các task.

Để chạy đúng mô hình “mỗi GPU là một stage process”, dùng `stage-server` trên từng
máy. Mỗi process chỉ load shard được chọn, không load toàn bộ model:

```bash
# GPU/máy giữ stage 0
MESHGPU_STAGE_CREDENTIAL="$STAGE0_SECRET" meshgpu stage-server \
  --manifest ./artifacts/tinyllama-1b-2s \
  --stage 0 --device cuda:0 --port 9100 \
  --cluster-id 1 --job-id 42 --lease-epoch 3 --worker-incarnation 101 \
  --tls-cert ./tls/stage0.crt --tls-key ./tls/stage0.key

# GPU/máy giữ stage 1: cùng manifest/config, chỉ cần shard stage 1 trên disk
MESHGPU_STAGE_CREDENTIAL="$STAGE1_SECRET" meshgpu stage-server \
  --manifest ./artifacts/tinyllama-1b-2s \
  --stage 1 --device cuda:0 --port 9101 \
  --cluster-id 1 --job-id 42 --lease-epoch 3 --worker-incarnation 202 \
  --tls-cert ./tls/stage1.crt --tls-key ./tls/stage1.key
```

`stage-server` là endpoint data-plane có auth và frame fencing; controller/launcher
phải cấp identity, credential, topology và tạo `StageRpcClient` tương ứng. Nếu bỏ
TLS, chỉ nên chạy trong mạng tin cậy để test local; production dùng `wss://` và
credential riêng theo job/stage.

### Hai Kaggle session qua outbound relay

Khi GPU 0 và GPU 1 nằm trong **hai Kaggle session khác nhau**, không dùng
`local_cuda` và không giả định notebook có cổng inbound công khai. Dựng một relay
trusted trên máy có địa chỉ `wss://` public; cả hai notebook và gateway đều mở
kết nối outbound tới relay. `stage-worker` chỉ tải shard của stage được giao:

```bash
# Máy relay — dùng chứng chỉ CA hợp lệ hoặc đặt sau reverse proxy TLS trên 443
MESHGPU_RELAY_TOKEN="$RELAY_SECRET" meshgpu relay-server \
  --host 0.0.0.0 --port 8443 \
  --tls-cert ./tls/relay.crt --tls-key ./tls/relay.key

# Kaggle session A — chỉ giữ stage 0 trên GPU của session A
MESHGPU_STAGE_CREDENTIAL="$STAGE0_SECRET" \
MESHGPU_RELAY_TOKEN="$RELAY_SECRET" meshgpu stage-worker \
  --manifest /kaggle/working/artifacts/qwen3-2s \
  --stage 0 --device cuda:0 \
  --relay-url wss://relay.example/v1/relay \
  --cluster-id 1 --job-id 42 --lease-epoch 3 \
  --worker-incarnation 101 --peer-worker-incarnation 301

# Kaggle session B — chỉ giữ stage 1 trên GPU của session B
MESHGPU_STAGE_CREDENTIAL="$STAGE1_SECRET" \
MESHGPU_RELAY_TOKEN="$RELAY_SECRET" meshgpu stage-worker \
  --manifest /kaggle/working/artifacts/qwen3-2s \
  --stage 1 --device cuda:0 \
  --relay-url wss://relay.example/v1/relay \
  --cluster-id 1 --job-id 42 --lease-epoch 3 \
  --worker-incarnation 102 --peer-worker-incarnation 301
```

Gateway dùng cùng route nhưng với `role=gateway`; URL nên được tạo bằng
`make_relay_url()` để tránh nhầm `job_id/stage_id`:

```python
from meshgpu.transport.relay import make_relay_url

stage_urls = [
    make_relay_url(
        "wss://relay.example/v1/relay",
        job_id=42,
        stage_id=stage_id,
        role="gateway",
    )
    for stage_id in (0, 1)
]
print(*stage_urls, sep="\n")
```

Sau đó truyền hai URL này vào `meshgpu remote-serve` (inference) hoặc
`meshgpu remote-ttt` (TTT), kèm `--relay-token` (hoặc
`MESHGPU_RELAY_TOKEN`) và credential tương ứng. Relay chỉ chuyển binary Stage-RPC; KV cache vẫn nằm ở
đúng Kaggle session sở hữu layer. Đây là đường **experimental**: Kaggle phải bật
Internet, phiên có thể bị thu hồi, và relay/controller phải nằm ngoài hai session.
Remote TTT hiện là đường correctness/experimental: stage cuối tính loss và giữ
autograd graph, còn gateway chỉ truyền activation/gradient boundary. Stage RPC có
memory gate trước khi gắn adapter và trước mỗi forward; nó trả
insufficient_memory thay vì cố chạy đến CUDA OOM. Nó chưa có checkpoint phân tán
hay automatic recovery giữa chừng.

Các template có sẵn trong `kaggle/meshgpu-stage0` và `kaggle/meshgpu-stage1`
dùng Kaggle CUDA image cố định, lấy source từ dataset đã pin, và đọc runtime
secret qua `kaggle_secrets.UserSecretsClient` (không nhúng token vào kernel).
Tạo ba secret trong **mỗi** session: `MESHGPU_RELAY_URL`,
`MESHGPU_RELAY_TOKEN`, và `MESHGPU_STAGE_CREDENTIAL`; credential stage 0/1
phải khác nhau. Hai file `.env_*` chỉ dùng để CLI xác thực khi
`kaggle kernels push/status/logs`, không được đưa vào notebook. Kernel sẽ dừng
ngay ở preflight nếu `torch.cuda.device_count() == 0`.

Có thể đặt một HTTP gateway ở máy điều phối. Gateway chỉ giữ metadata/tokenizer và
forward hidden state qua các endpoint stage:

```bash
meshgpu remote-serve \
  --manifest ./artifacts/tinyllama-1b-2s \
  --stage-url wss://gpu-a.example:9100 \
  --stage-url wss://gpu-b.example:9100 \
  --credential "$STAGE0_SECRET" --credential "$STAGE1_SECRET" \
  --stage-worker-incarnation 101 --stage-worker-incarnation 202 \
  --client-incarnation 301 --cluster-id 1 --job-id 42 --lease-epoch 3 \
  --tls-ca ./tls/cluster-ca.crt --port 8090
```

Các `--stage-url` phải theo thứ tự layer và số lượng phải khớp manifest. Một
`--credential` có thể dùng chung cho mọi stage; mặc định nên dùng credential riêng.
Nếu một stage không kết nối được, gateway đóng các kết nối đã mở và không khởi động
HTTP server nửa vời.

Trong deployment thật, mỗi stage endpoint phải load đúng shard và dùng identity/
credential do controller cấp; không mở endpoint RPC không có TLS/authentication.

## Cài đặt

```bash
python -m pip install -e '.[hf,data]'
```

Nếu chỉ chạy model bằng `prompt_ids`, không cần tokenizer. Để dùng prompt văn bản
hoặc fine-tune từ JSONL text, cài extra `hf` và giữ tokenizer trong artifact.

## Inference

### 1. Chia model Hugging Face

```bash
meshgpu convert TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --stages 2 \
  --dtype float16 \
  --revision <commit-or-tag> \
  --out ./artifacts/tinyllama-1b-2s
```

`MODEL_ID` cũng có thể là thư mục model local. Importer đọc indexed safetensors theo
từng stage để không cần materialize toàn bộ model lên một GPU. Hiện có hai adapter:
`llama_dense_v1` cho Llama dense giới hạn rõ ràng, và `qwen3_hf_v1` dùng trực tiếp
decoder block/cache chính thức của Transformers. Với Qwen3, ví dụ:

```bash
meshgpu convert Qwen/Qwen3-0.6B \
  --revision <commit-or-tag> --stages 2 --dtype bfloat16 \
  --attn-implementation sdpa --out ./artifacts/qwen3-0.6b-2s
```

Nên pin `--revision` khi tạo artifact. `--attn-implementation` được ghi vào
manifest; `flash_attention_2` chỉ dùng khi môi trường đã cài kernel tương ứng.

### 2. Chạy server

```bash
meshgpu serve \
  --manifest ./artifacts/tinyllama-1b-2s \
  --devices cuda:0,cuda:1 \
  --transport local_cuda \
  --port 8090
```

`--transport local_cuda` là đường ưu tiên cho hai GPU cùng máy: nếu CUDA báo peer
access thì boundary tensor vẫn ở CUDA; nếu không, MeshGPU dùng host staging rõ ràng.
`--transport cpu` là fallback tương thích rộng hơn và là lựa chọn an toàn cho stage
ở máy khác. Cả hai đều truyền hidden state giữa stage; KV cache không bị gửi qua
boundary.

Mỗi phần tử trong `--devices` tương ứng với một stage. Một device duy nhất sẽ được
lặp cho mọi stage, hữu ích khi test CPU:

```bash
meshgpu serve --manifest ./artifacts/tinyllama-1b-2s --device cpu
```

Gửi token IDs:

```bash
curl -N http://127.0.0.1:8090/v1/generate \
  -H 'content-type: application/json' \
  -d '{"prompt_ids":[1,42,17],"max_new_tokens":32,"stream":true,"seed":7}'
```

Nếu artifact có tokenizer, có thể gửi `prompt` thay cho `prompt_ids`. Mỗi request
chỉ được chọn một trong hai. `stream=true` trả SSE gồm token event và event `done`;
`stream=false` trả một JSON response.

## Training

Dataset là JSONL, mỗi dòng là một string hoặc object có field `text`:

```jsonl
{"text":"a short training example"}
{"text":"another example"}
```

Chạy LoRA (mặc định, phù hợp để bắt đầu):

```bash
meshgpu fine-tune \
  --manifest ./artifacts/tinyllama-1b-2s \
  --dataset ./data/train.jsonl \
  --out ./runs/llama-lora \
  --recipe lora \
  --devices cuda:0,cuda:1 \
  --steps 100 \
  --sequence-length 512 \
  --gradient-accumulation 4 \
  --transport local_cuda \
  --activation-checkpointing \
  --checkpoint-every 25
```

Artifact Qwen3 dùng chính lệnh này; `fine-tune` tự chọn config theo adapter trong
manifest và giữ đúng tokenizer/loss mask của artifact. `--recipe full` cập nhật
toàn bộ state trainable; `--recipe lora` chỉ cập nhật adapter. LoRA hiện là recipe
riêng, chưa phải QLoRA: đừng suy ra rằng quantizer + FSDP + kernel bất kỳ đều tương
thích.

Full fine-tuning dùng cùng lệnh với `--recipe full`; planner phải được chạy trước
vì optimizer state và gradient thường lớn hơn nhiều weight artifact:

```bash
meshgpu fine-tune ... --recipe full
```

Kết quả gồm:

- `runs/.../checkpoints/`: checkpoint resume, chỉ pointer đã commit mới được dùng;
- `runs/.../inference/manifest.json`: artifact đã merge LoRA (nếu có), sẵn sàng cho
  `meshgpu serve`;
- `runs/.../training_summary.json`: step cuối, checkpoint và đường artifact.

Có thể tiếp tục từ pointer checkpoint đã commit trong cùng thư mục output:

```bash
meshgpu fine-tune ... --out ./runs/llama-lora \
  --steps 200 --resume ckpt_00000100_ab12cd34
```

`--steps` là global step đích, không phải số step cộng thêm. Checkpoint training
lưu cursor theo packed batch và CLI sẽ bỏ qua đúng số batch đó khi resume; nếu
dataset, `--batch-size`, `--sequence-length` hoặc policy packing thay đổi thì
phải dùng một run/checkpoint mới, không coi đó là cùng một luồng dữ liệu.

Checkpoint mới lưu tên parameter theo đúng thứ tự từng optimizer group. Resume
từ chối checkpoint thiếu metadata này hoặc có ánh xạ khác với optimizer hiện tại;
checkpoint cũ đã commit vẫn có thể dùng để đọc/export weights, nhưng không tự suy
đoán ánh xạ optimizer state. Commit history được cập nhật dưới khóa file POSIX
(`flock`) và đọc lại từ đĩa để các coordinator dùng chung thư mục không ghi đè
lịch sử của nhau. Filesystem dùng chung phải hỗ trợ khóa POSIX và atomic rename.

## Lập kế hoạch VRAM

Planner không đoán VRAM của worker. Cần khai báo capacity thật hoặc map
`worker_specs`:

```yaml
task: inference
model:
  num_layers: 32
  hidden_size: 4096
  intermediate_size: 11008
  num_attention_heads: 32
  num_kv_heads: 8
  head_dim: 128
  vocab_size: 128256
  dtype_bytes: 2
placement:
  workers:
    - worker_id: gpu-a
      total_vram_gib: 24
    - worker_id: gpu-b
      total_vram_gib: 24
inference:
  max_prompt_tokens: 2048
  max_new_tokens: 256
```

```bash
meshgpu plan job.yaml
```

Với artifact đã convert, thay phần `model` bằng `manifest: ./artifact`. Report trả
layer range từng stage, peak weight/KV/activation/buffer, usable VRAM và bottleneck.

Để dùng số đo thực thay cho ước lượng, tạo `WorkloadSignature` tương ứng rồi lưu
`MemoryProfile`; `plan_from_profile(..., require_exact=True)` chỉ chọn các topology
đã có measurement đúng stage/range/workload. `require_exact=False` chỉ tạo preview,
và `as_preflight()` cố ý trả `feasible=false` nếu profile chưa an toàn cho admission.

Capacity acceptance có API trong `meshgpu.benchmarks.capacity`: cùng một
`CapacityContract` được đưa cho single-GPU runner, sharded runner và reference
runner. Kết quả chỉ là `passed` khi single runner OOM CUDA thật, sharded runner hoàn
tất, output khớp reference và có ít nhất hai CUDA device vật lý khác nhau. Thiếu
phần cứng trả `pending_hardware`, không giả lập thành công; peak allocated/reserved
được lưu theo từng device. Thí nghiệm cố ý làm OOM nên chạy mỗi runner ở process
riêng để không dùng lại CUDA context đã bị lỗi.

## Chạy kiểm chứng

```bash
ruff check src
python -m compileall -q src
python -m pytest -q
```

Correctness test so sánh pipeline với model Hugging Face nhỏ ở FP32; benchmark CLI:

```bash
python -m meshgpu.benchmarks.throughput --model tiny --stages 2 --device cpu --quick
```

## Kiến trúc ngắn gọn

```text
HF model
   │ convert
   ├── stage_00: embedding + layers [0, k)
   ├── stage_01: layers [k, m)
   └── stage_N : layers [...] + norm + lm_head
                         │
                 hidden state / frame transport
```

`native` dành cho integration PyTorch FSDP2/vLLM khi compatibility profile phù hợp.
`portable_pipeline` là backend reference có state ownership và recovery rõ ràng;
không giả định autograd tự đi xuyên network.

Các quyết định phạm vi model, boundary và capacity được ghi trong
[`docs/decisions/0001-model-adapter-and-local-gpu-boundary.md`](docs/decisions/0001-model-adapter-and-local-gpu-boundary.md).
