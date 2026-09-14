# MeshGPU Usage Guide

Tài liệu này hướng dẫn hai nhóm:

- **Người dùng/operator:** muốn convert model, chạy inference hoặc fine-tune.
- **Agent/automation:** muốn tự kiểm tra môi trường, chọn topology, khởi động worker,
  theo dõi job và xác nhận kết quả.

## 1. Mental model và giới hạn

MeshGPU chia một decoder model thành các stage liên tiếp. Mỗi GPU/process giữ một
phần model:

```text
stage 0: embedding + layers [0, k)
stage 1: layers [k, m)
stage N: layers [...] + norm + lm_head
```

Hidden state đi qua boundary giữa các stage; KV cache nằm tại stage sở hữu layer.
Hai GPU **không** trở thành một GPU có vùng VRAM liên tục. Mục tiêu đầu tiên là
chứa và chạy được model lớn hơn một card; thêm GPU không mặc nhiên làm inference
nhanh gấp đôi.

Các đường chạy hiện có:

| Workload | Topology | Trạng thái |
| --- | --- | --- |
| Inference một máy, nhiều GPU | `serve --transport local_cuda` | Đường ưu tiên |
| Inference process/host khác | `stage-server` hoặc `stage-worker` + RPC/relay | Đã có, cần benchmark |
| Inference hai Kaggle session | Outbound WSS relay | Experimental, đã smoke-test với Qwen3-0.6B |
| Fine-tune một máy, nhiều GPU | `fine-tune`, LoRA hoặc full | Đường chính hiện tại |
| Fine-tune qua WAN/Kaggle worker | Stage-RPC backward | Chưa hỗ trợ |

`local_cuda` chỉ dùng khi các stage ở cùng host. Stage ở host khác dùng `cpu`
transport qua RPC; không gửi CUDA pointer qua network.

## 2. Cài đặt và preflight

Từ root project:

```bash
python -m pip install -e '.[hf,data,dev]'
meshgpu doctor
```

`doctor` phải báo đúng PyTorch, CUDA, GPU count, VRAM và backend. Nếu dùng model
Hugging Face cần extra `hf`; nếu dùng YAML planner cần `data`.

Không bắt đầu load model khi preflight chưa qua. Đặc biệt:

- GPU phải xuất hiện trong `torch.cuda.device_count()`;
- dtype phải được GPU hỗ trợ;
- tổng `peak` của từng stage phải nhỏ hơn VRAM usable của chính GPU đó;
- prompt length, output budget và concurrency phải nằm trong admission budget.

## 3. Người dùng: inference local

### 3.1. Convert model Hugging Face

Pin revision để artifact reproducible. Qwen3 dùng adapter Hugging Face chính thức
và nên dùng SDPA; T4 thường bắt đầu với FP16.

```bash
meshgpu convert Qwen/Qwen3-0.6B \
  --revision c1899de289a04d12100db370d81485cdf75e47ca \
  --stages 2 \
  --dtype float16 \
  --attn-implementation sdpa \
  --out ./artifacts/qwen3-0.6b-2s
```

Llama dense cũng được hỗ trợ:

```bash
meshgpu convert TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --revision <commit-or-tag> \
  --stages 2 \
  --dtype float16 \
  --attn-implementation sdpa \
  --out ./artifacts/tinyllama-1b-2s
```

Artifact gồm `manifest.json`, config, và một file weight shard cho mỗi stage.
Importer đọc safetensors theo từng shard; runtime không cần đưa toàn bộ model vào
một GPU. Mặc định tokenizer cũng được copy vào artifact. Dùng `--no-tokenizer`
nếu caller chỉ gửi `prompt_ids`.

### 3.2. Kiểm tra placement trước khi chạy

Với job YAML:

```yaml
task: inference
model:
  num_layers: 28
  hidden_size: 1024
  intermediate_size: 3072
  num_attention_heads: 16
  num_kv_heads: 8
  head_dim: 128
  vocab_size: 151936
  dtype_bytes: 2
placement:
  workers:
    - worker_id: gpu-0
      total_vram_gib: 16
    - worker_id: gpu-1
      total_vram_gib: 16
inference:
  max_prompt_tokens: 2048
  max_new_tokens: 256
```

```bash
meshgpu plan job.yaml
```

Với artifact thật, dùng `manifest: ./artifacts/qwen3-0.6b-2s` trong spec để planner
đọc config, layer range và component size từ manifest. Nếu report `feasible=false`,
hãy giảm batch/concurrency hoặc đổi placement; không âm thầm cắt prompt.

### 3.3. Chạy server trên hai GPU cùng máy

```bash
meshgpu serve \
  --manifest ./artifacts/qwen3-0.6b-2s \
  --devices cuda:0,cuda:1 \
  --transport local_cuda \
  --max-prompt-tokens 2048 \
  --max-new-tokens 256 \
  --max-concurrent 4 \
  --port 8090
```

Nếu `local_cuda` không được GPU/driver hỗ trợ, dùng `--transport cpu` để ưu tiên
correctness. CPU cũng hữu ích làm reference, nhưng không phải benchmark hiệu năng.

Gửi token IDs:

```bash
curl -sS -N http://127.0.0.1:8090/v1/generate \
  -H 'content-type: application/json' \
  -d '{"prompt_ids":[1,42,17],"max_new_tokens":32,"stream":true,"seed":7}'
```

Với artifact có tokenizer, có thể gửi text:

```bash
curl -sS http://127.0.0.1:8090/v1/generate \
  -H 'content-type: application/json' \
  -d '{"prompt":"Explain model sharding in one sentence.","max_new_tokens":32,"stream":false}'
```

Một request chỉ gửi `prompt` **hoặc** `prompt_ids`. `stream=true` trả SSE token
events và event `done`; `stream=false` trả JSON một lần. KV cache được reserve và
release theo request; lỗi admission là lỗi có chủ đích, không phải cắt input.

### 3.4. Dùng Python API trực tiếp

Khi không cần HTTP server, pipeline có thể được gọi trực tiếp:

```python
import torch

from meshgpu.backends.portable.pipeline import (
    build_pipeline_from_manifest,
    pipeline_prefill,
)

# Artifact thật tự cung cấp config, layer ranges và shard weights.
workers = build_pipeline_from_manifest(
    "./artifacts/qwen3-0.6b-2s",
    devices=[torch.device("cuda:0"), torch.device("cuda:1")],
    transport="local_cuda",
)
logits = pipeline_prefill(
    workers,
    torch.tensor([[1, 42, 17]], dtype=torch.long),
    operation_id=1,
    attempt_id="request-1",
)
next_token = logits[:, -1, :].argmax(dim=-1)
print(next_token.tolist())
```

Đối với artifact Qwen, dùng `build_pipeline_from_manifest()` để runtime lấy đúng
Qwen config/adapter. Không tự thay config Llama cho Qwen.

## 4. Người dùng: fine-tune

Fine-tune hiện chạy trong một process/pipeline cố định trên cùng host. Dataset là
JSONL, mỗi dòng là string hoặc object có field `text`:

```jsonl
{"text":"A short training example."}
{"text":"Another training example."}
```

### 4.1. LoRA — lựa chọn bắt đầu

```bash
meshgpu fine-tune \
  --manifest ./artifacts/qwen3-0.6b-2s \
  --dataset ./data/train.jsonl \
  --out ./runs/qwen3-lora \
  --recipe lora \
  --devices cuda:0,cuda:1 \
  --transport local_cuda \
  --steps 100 \
  --batch-size 1 \
  --sequence-length 512 \
  --gradient-accumulation 4 \
  --lora-rank 16 \
  --lora-alpha 32 \
  --activation-checkpointing \
  --checkpoint-every 25
```

LoRA freeze base weights và chỉ cập nhật adapter. Đây là LoRA recipe, không tự động
đồng nghĩa với QLoRA/4-bit quantization.

### 4.2. Full fine-tuning

```bash
meshgpu fine-tune \
  --manifest ./artifacts/qwen3-0.6b-2s \
  --dataset ./data/train.jsonl \
  --out ./runs/qwen3-full \
  --recipe full \
  --devices cuda:0,cuda:1 \
  --transport local_cuda \
  --steps 100 \
  --sequence-length 512 \
  --gradient-accumulation 4 \
  --activation-checkpointing
```

Full training cần ngân sách lớn hơn nhiều vì có gradient và optimizer state. Luôn
chạy CUDA training preflight trước khi load weight; nếu bị từ chối thì đó là hành vi
đúng.

### 4.3. Resume và export

Kết quả nằm trong `runs/...`:

- `checkpoints/`: shard checkpoint và committed pointer;
- `inference/manifest.json`: artifact inference sau khi merge LoRA nếu cần;
- `training_summary.json`: step, checkpoint và artifact cuối.

Resume dùng checkpoint ID đã committed:

```bash
meshgpu fine-tune \
  --manifest ./artifacts/qwen3-0.6b-2s \
  --dataset ./data/train.jsonl \
  --out ./runs/qwen3-lora \
  --recipe lora \
  --devices cuda:0,cuda:1 \
  --steps 200 \
  --resume ckpt_00000100_ab12cd34
```

`--steps` là global step đích, không phải số step cộng thêm. Không resume cùng run
nếu dataset, batch size, sequence length hoặc packing policy đã đổi.

## 5. Hai Kaggle session khác nhau qua relay

Đây là flow để GPU ở **hai session Kaggle độc lập** cùng chạy một model. Mỗi session
chỉ giữ một stage; không giả định session có inbound port public.

### 5.1. Chuẩn bị relay

Production nên dùng relay sau reverse proxy TLS hoặc relay tự có certificate:

```bash
export RELAY_TOKEN='<lưu trong secret manager, không commit>'
MESHGPU_RELAY_TOKEN="$RELAY_TOKEN" meshgpu relay-server \
  --host 0.0.0.0 \
  --port 8443 \
  --tls-cert ./tls/relay.crt \
  --tls-key ./tls/relay.key
```

Để smoke-test nhanh trên máy dev, có thể chạy relay `ws://` phía sau Cloudflare
Quick Tunnel. Cloudflare cung cấp URL public `https://...trycloudflare.com`; worker
và gateway dùng cùng host với scheme `wss://`:

```bash
MESHGPU_RELAY_TOKEN="$RELAY_TOKEN" meshgpu relay-server \
  --host 0.0.0.0 --port 8765
cloudflared tunnel --url http://127.0.0.1:8765
```

Quick Tunnel chỉ dành cho smoke-test; nó không phải SLA hay multi-tenant broker.

### 5.2. Secret contract trong mỗi Kaggle session

Trong **cả hai** session, tạo Kaggle Secrets với đúng tên:

```text
MESHGPU_RELAY_URL       = wss://<relay-host>/v1/relay
MESHGPU_RELAY_TOKEN     = <shared relay token>
MESHGPU_STAGE_CREDENTIAL= <credential riêng của stage đó>
```

Stage 0 và stage 1 phải có credential khác nhau. Template đọc các secret này bằng
`kaggle_secrets.UserSecretsClient`; `.env_BnhAnh` và `.env_beo` chỉ chứa
`KAGGLE_API_TOKEN` để CLI xác thực, không được upload vào kernel.

CLI Kaggle hiện không có lệnh portable để set User Secrets; set chúng trong giao
diện Secrets của Kaggle hoặc secret mechanism được tổ chức quản lý. Không thay bằng
literal trong `meshgpu-stage0.py`/`meshgpu-stage1.py`.

### 5.3. Push hai kernel bằng hai credential

Template đã có sẵn:

- `kaggle/meshgpu-stage0`: session A, stage 0, worker incarnation `1001`;
- `kaggle/meshgpu-stage1`: session B, stage 1, worker incarnation `1002`.

Push bằng đúng file env tương ứng:

```bash
# Account/session A — stage 0
set -a
. ./.env_BnhAnh
set +a
kaggle kernels push \
  -p kaggle/meshgpu-stage0 \
  --timeout 1800 \
  --accelerator NvidiaTeslaT4

# Account/session B — stage 1
set -a
. ./.env_beo
set +a
kaggle kernels push \
  -p kaggle/meshgpu-stage1 \
  --timeout 1800 \
  --accelerator NvidiaTeslaT4
```

Template hiện dùng CUDA image cố định, Internet bật, source dataset
`bnhanh/meshgpu-source-20260914`, và convert Qwen3-0.6B ở revision đã pin. Nếu đổi
model hoặc revision, phải tạo artifact/source version mới và kiểm tra lại manifest.

Theo dõi bằng chính credential tương ứng:

```bash
set -a; . ./.env_BnhAnh; set +a
kaggle kernels status bnhanh/meshgpu-wan-stage-0-kaggle
kaggle kernels logs bnhanh/meshgpu-wan-stage-0-kaggle

set -a; . ./.env_beo; set +a
kaggle kernels status nituv05/meshgpu-wan-stage-1-kaggle
kaggle kernels logs nituv05/meshgpu-wan-stage-1-kaggle
```

Preflight phải thấy `torch.cuda.is_available()=True`, số GPU dương và stage phải log
đã load trên `cuda:0`. `RUNNING` một mình chưa phải bằng chứng worker đã nối relay.

### 5.4. Mở gateway và gọi inference

Gateway cần bản artifact local có `manifest.json`; vì gateway không giữ weight nên
chỉ cần metadata/config nếu dùng `prompt_ids`.

```bash
export RELAY_BASE='wss://<relay-host>/v1/relay'
export RELAY_TOKEN='<relay token>'
export STAGE0_SECRET='<stage 0 credential>'
export STAGE1_SECRET='<stage 1 credential>'

STAGE0_URL="${RELAY_BASE}?job_id=4242&stage_id=0&role=gateway"
STAGE1_URL="${RELAY_BASE}?job_id=4242&stage_id=1&role=gateway"

meshgpu remote-serve \
  --manifest ./artifacts/qwen3-0.6b-2s \
  --stage-url "$STAGE0_URL" \
  --stage-url "$STAGE1_URL" \
  --credential "$STAGE0_SECRET" \
  --credential "$STAGE1_SECRET" \
  --relay-token "$RELAY_TOKEN" \
  --stage-worker-incarnation 1001 \
  --stage-worker-incarnation 1002 \
  --client-incarnation 3001 \
  --cluster-id 7 \
  --job-id 4242 \
  --lease-epoch 1 \
  --port 8090
```

URL phải giữ đúng `job_id`, `stage_id` và `role=gateway`; token nằm trong HTTP
header, không nằm trong URL. Với private CA, thêm `--tls-ca`; với public CA của
Cloudflare/system trust store thì không cần.

Gọi gateway bằng token IDs:

```bash
curl -sS -N http://127.0.0.1:8090/v1/generate \
  -H 'content-type: application/json' \
  -d '{"prompt_ids":[1,4,9],"max_new_tokens":8,"stream":true}'
```

Kết quả nghiệm thu tối thiểu phải có:

1. Hai kernel đều có CUDA và log stage loaded.
2. Gateway log `connected stages=2`.
3. Prefill trả logits có shape đúng vocab size.
4. Decode trả token và KV length của hai stage bằng nhau.
5. Sau khi request kết thúc, KV được clear hoặc release; không tăng vô hạn qua nhiều request.

Đường Kaggle/WAN là experimental. Smoke-test Qwen3-0.6B hai session đã chạy được
forward thật, nhưng latency relay có thể rất cao; không dùng kết quả smoke đó làm
throughput claim.

## 6. Người dùng: stage RPC trực tiếp

Khi host tự quản lý và có route inbound, chạy một RPC server cho mỗi stage:

```bash
# Stage 0
MESHGPU_STAGE_CREDENTIAL="$STAGE0_SECRET" meshgpu stage-server \
  --manifest ./artifacts/qwen3-0.6b-2s \
  --stage 0 --device cuda:0 --port 9100 \
  --cluster-id 7 --job-id 4242 --lease-epoch 1 \
  --worker-incarnation 1001 --peer-worker-incarnation 3001 \
  --tls-cert ./tls/stage0.crt --tls-key ./tls/stage0.key

# Stage 1
MESHGPU_STAGE_CREDENTIAL="$STAGE1_SECRET" meshgpu stage-server \
  --manifest ./artifacts/qwen3-0.6b-2s \
  --stage 1 --device cuda:0 --port 9101 \
  --cluster-id 7 --job-id 4242 --lease-epoch 1 \
  --worker-incarnation 1002 --peer-worker-incarnation 3001 \
  --tls-cert ./tls/stage1.crt --tls-key ./tls/stage1.key
```

Gateway khi đó dùng:

```bash
meshgpu remote-serve \
  --manifest ./artifacts/qwen3-0.6b-2s \
  --stage-url wss://gpu-a.example:9100 \
  --stage-url wss://gpu-b.example:9101 \
  --credential "$STAGE0_SECRET" \
  --credential "$STAGE1_SECRET" \
  --stage-worker-incarnation 1001 \
  --stage-worker-incarnation 1002 \
  --client-incarnation 3001 \
  --cluster-id 7 --job-id 4242 --lease-epoch 1 \
  --tls-ca ./tls/cluster-ca.crt
```

Mỗi `--stage-url` phải theo đúng thứ tự layer trong manifest. Không mở stage RPC
public nếu thiếu TLS và credential.

## 7. Agent/automation runbook

Agent không tự suy ra rằng “cộng VRAM là đủ”. Agent phải thực hiện các phase dưới
đây và lưu lại output/decision.

### Phase A — nhận input và preflight

Input tối thiểu:

```text
mode: inference | lora | full
model_id hoặc artifact_dir
revision (nếu convert từ Hub)
stage_count và device/provider list
prompt/output budget hoặc training sequence/batch budget
transport policy: local_cuda | cpu | relay
```

Checklist:

1. Chạy `meshgpu doctor` trên từng worker.
2. Gọi `detect_provider()`/`check_eligibility()`; Kaggle là `experimental`, Colab
   managed free không được làm distributed worker.
3. Xác nhận Internet/outbound relay nếu provider cần.
4. Đọc và verify `manifest.json`, config, shard size/hash.
5. Chạy planner theo đúng workload (`inference` khác `training`).
6. Nếu không fit, trả `insufficient_memory` kèm stage bottleneck và lý do; không
   đổi model/precision âm thầm và không cắt input.

### Phase B — chuẩn bị artifact và placement

1. Pin model revision, dtype và attention implementation.
2. Chọn layer ranges contiguous; stage cuối phải tính cả norm/lm_head, stage đầu
   phải tính embedding.
3. Ghi lại mapping `stage_id → worker_id → device → [layer_start, layer_end)`.
4. Với cùng host, ưu tiên `local_cuda`; với host khác dùng TLS RPC; với Kaggle dùng
   outbound WSS relay.
5. Không load toàn bộ model vào mọi worker để “test cho nhanh”; worker phải load
   shard được giao.

### Phase C — khởi động và fencing

Agent tạo một identity nhất quán cho job:

| Trường | Ý nghĩa |
| --- | --- |
| `cluster_id` | cluster logical |
| `job_id` | job inference/training |
| `lease_epoch` | phiên lease hiện tại |
| `worker_incarnation` | mỗi process restart phải tăng/đổi |
| `peer_worker_incarnation` | incarnation peer được phép nhận |
| `operation_id`/`attempt_id` | một operation và lần thử của nó |

Không reuse `worker_incarnation` sau restart. Khi topology/lease đổi, không để frame
cũ tiếp tục ghi vào KV hoặc optimizer state.

### Phase D — verify trước workload thật

Agent phải kiểm tra theo thứ tự:

```text
worker registered/loaded
→ transport authenticated
→ all stage routes paired
→ clear/health RPC
→ one short prefill
→ one decode
→ KV lengths equal
→ only then accept user workload
```

`RUNNING`, `connected TCP`, hoặc HTTP health của controller chưa đủ để kết luận
inference thành công. Phải có logits/token từ pipeline.

### Phase E — inference loop

Agent giữ session state gồm prompt prefix, token đã xác nhận, sampling state, route
version và cache key. Mỗi decode chỉ gửi boundary hidden/token cần thiết; không gửi
toàn bộ KV cache qua network.

Nếu request bị admission reject, trả lỗi rõ ràng. Nếu output stream đã gửi token,
chỉ đánh dấu token là confirmed sau khi operation/sequence được xác nhận; không gửi
lại token mù quáng khi reconnect.

### Phase F — training loop

Agent training phải:

1. chạy capacity preflight cho weight + activation + gradient + optimizer + comm buffer;
2. tạo optimizer theo trainable params của từng stage;
3. ghi global step, data cursor, gradient accumulation và checkpoint ID;
4. checkpoint định kỳ, verify hash, chỉ resume từ committed pointer;
5. sau training, export artifact inference và kiểm tra load lại.

Stage-RPC hiện chỉ mở inference. Không khởi động `fine-tune` với hai Kaggle
`stage-worker` rồi kỳ vọng backward tự đi qua socket.

### Phase G — recovery và cleanup

Khi worker mất:

1. dừng nhận work mới cho route lỗi;
2. tăng incarnation và cấp lease/attempt mới;
3. restore weight/optimizer từ checkpoint nếu là training;
4. inference phải replay **confirmed prefix** để dựng lại KV;
5. tiếp tục decode từ `confirmed_seq_num`, không replay token chưa xác nhận;
6. nếu không chứng minh state nhất quán, fail request thay vì retry mù.

API replay hiện có:

```python
from meshgpu.inference.recovery import replay_prefix_async, resume_session

result = await replay_prefix_async(
    workers,
    prefix_ids,
    operation_id=next_operation_id,
    attempt_id=new_attempt_id,
    cache_key=f"session:{session_id}",
)
session = await resume_session(
    workers,
    session_id,
    prefix_ids.tolist()[0],
    max_new_tokens,
    sampling,
    confirmed_seq_num=last_confirmed_seq,
)
```

Cleanup gồm close RPC clients, clear KV, cancel server/relay process và dừng/delete
Kaggle smoke kernel để không đốt quota. Không xóa checkpoint committed trước khi
đã có bản backup/manifest thay thế.

## 8. Diagnostics nhanh

| Triệu chứng | Kiểm tra | Cách xử lý |
| --- | --- | --- |
| `cuda_available=False` | Kaggle metadata, `doctor`, `CUDA_VISIBLE_DEVICES` | Bật GPU runtime đúng machine shape; không tiếp tục load |
| `insufficient_memory` | planner summary, free VRAM, sequence/concurrency | Giảm budget hoặc đổi placement; không cắt prompt |
| `missing Kaggle Secret` | Ba tên `MESHGPU_*` trong session | Tạo secret trong Kaggle; không hard-code vào script |
| `relay authentication failed` | relay token header và URL host/path | So khớp token, `/v1/relay`, role và job |
| `ssl=None is incompatible with a wss:// URI` | source dataset/package version | Dùng source hiện tại có default SSL context |
| `keepalive ping timeout` | relay/WAN và cold GPU latency | Dùng source hiện tại; ping timeout WAN đã tăng, vẫn đo lại link |
| `could not connect all stage RPC clients` | từng stage URL/credential/incarnation | Kiểm tra route pair và logs từng kernel |
| KV lengths khác nhau | stage restart hoặc request cleanup lỗi | clear/replay confirmed prefix; không decode tiếp với cache lệch |
| output đúng shape nhưng token sai | revision/config/dtype/attention/reference | So sánh với Hugging Face reference ở cùng precision và input |
| fine-tune OOM | optimizer/activation/sequence budget | bật checkpointing, giảm batch/sequence, hoặc dùng LoRA |

Khi debug, ghi error type, job/stage/operation ID và metric; không ghi relay token,
stage credential, API token hoặc toàn bộ prompt nhạy cảm vào log.

## 9. Acceptance checklist

### Local inference

```bash
meshgpu doctor
meshgpu plan job.yaml
meshgpu serve --manifest ./artifacts/model-2s \
  --devices cuda:0,cuda:1 --transport local_cuda
```

Đạt khi output sharded khớp reference, mọi stage fit VRAM, KV không tăng bất thường
qua nhiều request và peak VRAM từng GPU được ghi lại.

### Training

Đạt khi LoRA/full loss chạy, adapter/base param update đúng recipe, checkpoint
resume cho cùng data cursor, export load được, và inference sau export khớp artifact
trước export trong tolerance đã chọn.

### Hai Kaggle session

Đạt khi có bằng chứng đồng thời từ hai credential:

```text
CUDA preflight A + stage 0 loaded
CUDA preflight B + stage 1 loaded
gateway connected stages=2
prefill logits
decode token
equal KV lengths
```

Không coi chỉ có `kaggle kernels status=RUNNING` là pass.

## 10. Quality gate trước khi gửi review

```bash
ruff check src tests kaggle
python -m compileall -q src kaggle
python -m pytest -q
```

Khi sửa transport/protocol, phải chạy thêm:

```bash
python -m pytest \
  tests/unit/test_connection.py \
  tests/unit/test_relay.py \
  tests/integration/test_stage_rpc.py \
  tests/integration/test_remote_gateway.py -q
```

Các thay đổi liên quan provider/Kaggle phải test preflight trên provider thật.
Local unit test xanh không thay thế hardware acceptance.

## 11. Lệnh CLI tham khảo

```bash
meshgpu --help
meshgpu doctor
meshgpu convert --help
meshgpu plan job.yaml
meshgpu serve --help
meshgpu fine-tune --help
meshgpu stage-server --help
meshgpu stage-worker --help
meshgpu relay-server --help
meshgpu remote-serve --help
meshgpu benchmark --help
```

Controller/agent lifecycle cũng có sẵn cho deployment có control plane:

```bash
meshgpu controller --host 0.0.0.0 --port 8080 --db meshgpu.db
MESHGPU_CONTROLLER=http://controller:8080 \
MESHGPU_JOIN_TOKEN="$JOIN_TOKEN" meshgpu agent
MESHGPU_CONTROLLER=http://controller:8080 meshgpu workers list
MESHGPU_CONTROLLER=http://controller:8080 meshgpu jobs status <job-id>
MESHGPU_CONTROLLER=http://controller:8080 meshgpu jobs cancel <job-id>
```

Controller điều phối lease/lifecycle; nó không thay thế capacity preflight và cũng
không làm cho worker Kaggle có inbound network.
