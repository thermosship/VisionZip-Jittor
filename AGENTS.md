# VisionZip-Jittor Codex Handoff and Project State

> **Purpose:** This is the authoritative cross-account handoff file for Codex agents working on this reproduction. A new agent may have no access to earlier chats. Read this file completely before modifying code, running expensive jobs, changing claims, or proposing the next phase.
>
> **Last authoritative update:** 2026-08-03 (Asia/Shanghai), after passwordless VS Code/SSH validation and before Phase 4A implementation.
>
> **Current phase boundary:** Phase 1, Phase 2, Phase 3A, and the minimum real-LLM Phase 3B are complete. Phase 4A real Projector training has **not started**.

## 1. Mandatory instructions for every future Codex agent

1. Communicate with the user in Chinese.
2. When the user must run commands manually, label them explicitly as either:
   - **Windows PowerShell**; or
   - **AutoDL/Jupyter Terminal**.
3. Do not ask for, print, store, or commit the user's AutoDL password, tokens, private keys, or other secrets.
4. Before any work, verify the live state instead of assuming this file is perfectly current:

   ```bash
   cd /root/autodl-tmp/VisionZip-Jittor
   git status --short --branch
   git log -6 --oneline
   nvidia-smi
   free -h
   ```

5. After every meaningful code change, experiment, benchmark, result archive, or phase transition, update this file in the same work session. At minimum update:
   - `Last authoritative update`;
   - `Current live state`;
   - the relevant phase/result section;
   - `Next exact actions`;
   - `Handoff update log`.
6. Preserve historical measured values. Do not silently replace old results with new runs. Add a dated result and explain why it supersedes or differs from an older result.
7. Do not weaken acceptance criteria merely to make a run pass.
8. Do not claim captioning/VQA quality from random-Projector smoke output. Phase 3B proves execution and gradient isolation, not trained visual-language quality.
9. Do not run `git add .` or `git add -A` while generated artifacts are present. The remote tree contains a roughly 475 MiB GPT-2 NPZ and other evidence outputs. Stage source/document files explicitly.
10. Keep PyTorch export and Jittor execution environments isolated. The `USE_TORCH` environment variable is a known trap; see the environment section below.
11. Use `tmux` for long training or download jobs so a VS Code/SSH disconnect does not terminate the run.
12. Prefer reproducible scripts, JSON summaries, hashes, and committed documentation over results that exist only in terminal scrollback.
13. GitHub and Hugging Face connectivity from AutoDL can be unstable. Do not interpret a network failure as a code failure. Proven fallbacks are network turbo, `HF_HUB_DISABLE_XET=1`, resumable cache use, and Git Bundle transfer.

## 2. User, repository, and access context

### Repository locations

```text
GitHub: https://github.com/thermosship/VisionZip-Jittor
Windows working copy: C:\Users\69444\Desktop\cmm\VisionZip-Jittor
AutoDL working copy: /root/autodl-tmp/VisionZip-Jittor
AutoDL Hugging Face cache: /root/autodl-tmp/cache/huggingface
Jittor environment: /root/autodl-tmp/envs/visionzip-jittor
```

### Passwordless remote access

The Windows SSH alias is:

```text
autodl-visionzip
```

The host and port are intentionally not recorded here because AutoDL instance endpoints can change. They are stored in the user's local file:

```text
C:\Users\69444\.ssh\config
```

Validated Windows command:

```powershell
ssh -o BatchMode=yes autodl-visionzip "echo SSH_OK; hostname; pwd"
```

Validated output on 2026-08-03:

```text
SSH_OK
autodl-container-10894aa74d-da4e9cbe
/root
```

Codex may now operate the remote repository through passwordless SSH. A VS Code Remote-SSH window and SSH commands address the same files under `/root/autodl-tmp/VisionZip-Jittor`.

### Resource ownership under VS Code Remote-SSH

- Local Windows CPU/RAM: VS Code UI, SSH transport, terminal rendering.
- AutoDL CPU/RAM: Python, Jittor, PyTorch, preprocessing, NPZ loading, VS Code Server.
- AutoDL GPU VRAM: CUDA tensors and model execution.
- AutoDL disk: source, caches, weights, logs, checkpoints, and outputs.

The user's low local RAM does not limit remote model loading. Avoid opening large NPZ files directly in the VS Code editor.

## 3. Current live state

Verified on 2026-08-03 immediately before creating this file:

```text
Remote host: autodl-container-10894aa74d-da4e9cbe
Repository: /root/autodl-tmp/VisionZip-Jittor
Branch: main
Upstream state: main is up to date with origin/main
HEAD before this handoff file: c2ac559 docs: record phase-three real GPT-2 results
GPU: NVIDIA GeForce RTX 4090, 24564 MiB
GPU state: idle, 0 MiB used at verification time
Driver: 580.105.08
nvidia-smi maximum CUDA compatibility: 13.0
Jittor nvcc toolkit: 11.8.89
Remote RAM: approximately 1.0 TiB total, approximately 904 GiB available at verification time
Remote data disk: /root/autodl-tmp, 200 GiB total, approximately 197 GiB available
```

Current untracked generated paths:

```text
logs/benchmark_gpu_info.txt
logs/phase3b/
outputs/phase3b/
```

These are not pending source changes. In particular:

```text
outputs/phase3b/gpt2/gpt2_float32_weights.npz  ~475 MiB
```

Do not commit these generated artifacts to ordinary Git history.

## 4. Environment contract

### 4.1 Native Jittor environment

Use for native VisionZip, Projector, GPT-2 execution, training, and Jittor tests:

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/envs/visionzip-jittor
export OMP_NUM_THREADS=8
```

Verified:

```text
Python: 3.10.20
Python path: /root/autodl-tmp/envs/visionzip-jittor/bin/python
Jittor: 1.3.11.0
CUDA architecture: sm_89
nvcc: 11.8.89
```

For the native Phase 3B runner, framework backends must remain disabled before importing Transformers tokenizer utilities:

```text
USE_TORCH=0
USE_TF=0
USE_FLAX=0
```

`scripts/run_phase3b_gpt2.py` sets these values explicitly.

### 4.2 PyTorch reference/export environment

Use the explicit interpreter:

```text
/root/miniconda3/bin/python
```

Verified reference versions:

```text
PyTorch: 2.1.2+cu118
Transformers: 4.31.0
CUDA available: true
GPU: NVIDIA GeForce RTX 4090
```

For PyTorch/Hugging Face exporters, explicitly enable PyTorch because a shell may retain `USE_TORCH=0` from the Jittor runner:

```bash
export USE_TORCH=1
export USE_TF=0
export USE_FLAX=0
```

Commit `f6a346d` made this isolation explicit in the scripts.

### 4.3 Hugging Face network settings proven on AutoDL

```bash
source /etc/network_turbo
export HF_HUB_DISABLE_XET=1
export HF_HUB_ETAG_TIMEOUT=60
export HF_HUB_DOWNLOAD_TIMEOUT=600
export HF_HOME=/root/autodl-tmp/cache/huggingface
export TRANSFORMERS_CACHE=/root/autodl-tmp/cache/huggingface/transformers
```

Why:

- direct Hugging Face access previously timed out;
- Xet/CAS previously returned `401 Unauthorized`;
- `HF_HUB_DISABLE_XET=1` allowed the CLIP and GPT-2 artifacts to download normally;
- use the data-disk cache instead of filling the system disk.

## 5. Reproduction objective and fixed technical scope

The project reproduces the VisionZip token compression path in native Jittor and aligns it with a PyTorch reference derived from the official implementation.

Fixed upstream reference information:

```text
Official upstream commit: 8f86b55c6f000eb033e6912538af2dd7dcb30502
Vision encoder: openai/clip-vit-large-patch14-336
CLIP input resolution: 336 x 336
Patch grid: 24 x 24
Input sequence: 577 tokens = 1 CLS + 576 patch tokens
Feature width: 1024
Metric width: 64
CLIP layer: layer_index=22, hidden_states_index=23, attention_index=22
Metric source: k_proj -> [B,H,N,D] -> mean(heads)
Default merge mode: code_exact
Precision used for formal alignment: float32
Seed: 2026
```

Token budget convention:

| Nominal budget | Dominant patch tokens | Contextual tokens | CLS additional | Actual output |
|---:|---:|---:|---:|---:|
| 64 | 54 | 10 | 1 | 65 |
| 128 | 108 | 20 | 1 | 129 |
| 192 | 162 | 30 | 1 | 193 |

Important semantic rule: the nominal budget excludes CLS. Do not report 64 as an actual 64-token output; it is 65 including CLS.

`code_exact` follows official implementation behavior. `paper_avg` exists only as a distinct ablation and must not be confused with the default official-code-aligned path.

## 6. Phase history and validated results

### 6.1 Phase 1 — native Jittor core alignment: COMPLETE

Purpose:

- implement Dominant Token Selection and Contextual Token Merging in native Jittor;
- compare against literal PyTorch behavior using deterministic synthetic full-shape inputs;
- benchmark the compression core only.

Milestone commits:

```text
4692372 feat: implement phase-one VisionZip Jittor core alignment
04106cc docs: record phase-one alignment results
1261c42 docs: add RTX 4090 core benchmark results
```

Formal input shapes:

```text
hidden_states: [1,577,1024]
attentions: [1,16,577,577]
metric: [1,577,64]
```

All 64/128/192 budgets passed at `atol=1e-5, rtol=1e-5`:

| Budget | Actual output | Compressed max abs | Contextual max abs | Index agreement | Assignment agreement |
|---:|---:|---:|---:|---:|---:|
| 64 | 65 | 2.3841858e-07 | 2.3841858e-07 | 100% | 100% |
| 128 | 129 | 2.3841858e-07 | 2.3841858e-07 | 100% | 100% |
| 192 | 193 | 2.3841858e-07 | 2.3841858e-07 | 100% | 100% |

RTX 4090 core-only measurements, batch 1, FP32, 20 warmups, 100 iterations:

| Budget | Mean ms | Median ms | Calls/s |
|---:|---:|---:|---:|
| 64 | 0.977824 | 0.919659 | 1022.68 |
| 128 | 0.924812 | 0.870302 | 1081.30 |
| 192 | 0.943357 | 0.892080 | 1059.00 |

Interpretation boundary: these numbers measure the VisionZip compression core only, not CLIP encoding, LLM prefill, or end-to-end generation.

Detailed document: `docs/PHASE1_RESULTS.md`.

### 6.2 Phase 2 — real CLIP feature alignment: COMPLETE

Purpose:

- extract real hidden states, attentions, and key-projection metrics from CLIP ViT-L/14-336;
- feed identical arrays to PyTorch and Jittor VisionZip;
- validate three budgets and produce token visualizations.

Milestone commits:

```text
b5eb4c1 feat: add real CLIP feature alignment pipeline
df23929 fix: match PyTorch CUDA normalization for CLIP metrics
c75af09 docs: record phase-two real CLIP alignment results
```

Formal batch:

```text
assets/sample_images/dense.png
assets/sample_images/scene.png
assets/sample_images/text.png
pixel_values: [3,3,336,336]
hidden_states: [3,577,1024]
attentions: [3,16,577,577]
metric: [3,577,64]
```

Results:

| Budget | Output | Compressed max abs | Contextual max abs | Assignment | Result |
|---:|---|---:|---:|---:|---|
| 64 | `[3,65,1024]` | 5.7220458984375e-06 | 5.7220458984375e-06 | exact 1.0 | PASS |
| 128 | `[3,129,1024]` | 1.9073486328125e-06 | 1.9073486328125e-06 | exact 1.0 | PASS |
| 192 | `[3,193,1024]` | 1.9073486328125e-06 | 1.9073486328125e-06 | exact 1.0 | PASS |

All relevant floating arrays passed `atol=1e-5, rtol=1e-5`. All discrete indices and assignments were exactly equal. The pipeline summary recorded `passed: true`. Nine visualization PNGs were generated.

#### Near-tie CUDA normalization issue

Initial Jittor normalization produced four assignment mismatches. Numeric-path isolation showed:

```text
Jittor norm + Jittor BMM: 4 mismatches
PyTorch norm + Jittor BMM: 3 mismatches
PyTorch normalized operands + Jittor BMM: 0 mismatches
```

The final fix is `_torch_cuda_l2_normalize_64` in `visionzip_jittor/core.py`. For CUDA FP32 `[B,N,64]` with `eps=0.0`, it uses a custom `jt.code` CUDA kernel with PyTorch-compatible reduction order and explicit round-to-nearest operations such as `__fmul_rn`, `__fadd_rn`, `__fsqrt_rn`, and `__fdiv_rn`. The diagnostic then achieved exact norm, normalized values, similarity, and assignments.

Do not replace this path with a generic reduction without rerunning the real-CLIP near-tie diagnostics.

Evidence archive:

```text
File: VisionZip-Jittor-phase2-evidence-20260802.tar.gz
SHA256: 8886E0FE914A0D68AEC70346005853DC83A9086185D58DCFE945D040DE612CDC
Entries: 35
Visualization PNGs: 9
```

Detailed document: `docs/PHASE2_REAL_CLIP.md`.

### 6.3 Phase 3A — Projector plus frozen language stub: COMPLETE

Purpose:

- prove that real Phase 2 CLIP features flow through native Jittor VisionZip and a trainable Projector;
- prove Projector-only optimization and frozen-language gradient isolation before implementing a real LLM.

Milestone commits:

```text
a8451e8 feat: add phase-three projector smoke integration
f10b905 docs: record phase-three projector smoke results
```

Configuration:

```text
Projector: mlp2x_gelu, 1024 -> 4096 -> 4096
Projector parameters: 20,979,712
Frozen language stub parameters: 1,048,576
Optimizer: Adam, Projector parameters only
real_llm: false
```

All 64/128/192 budgets passed. Each had exact Phase 2 assignments, allclose compressed features, four Projector tensors with finite nonzero gradients, changed Projector weights, unchanged frozen-language weights, and top-level `passed: true`.

Evidence archive transferred to Windows:

```text
File: VisionZip-Jittor-phase3a-evidence-20260802.tar.gz
SHA256: 42046F581613901BD2A8395EE59B086B7B35419C330E0DA7FE361777CE15C69F
```

Interpretation boundary: Phase 3A uses a language surrogate and is not real LLM validation.

Detailed document: `docs/PHASE3_PROJECTOR_FROZEN_LLM.md`.

### 6.4 Phase 3B — native Jittor real frozen GPT-2: COMPLETE

Purpose:

- replace the stub with a real GPT-2 small implementation in native Jittor;
- export Hugging Face GPT-2 weights and tokenizer artifacts;
- align text-only logits;
- execute real causal loss, Projector-only backward/update, and greedy generation for all budgets.

Milestone commits:

```text
c1b8244 feat: add phase-three real GPT-2 integration
f6a346d fix: isolate Phase 3B framework environment flags
c2ac559 docs: record phase-three real GPT-2 results
```

Architecture:

```text
Real CLIP Phase 2 features
-> native Jittor VisionZip
-> trainable Jittor Projector 1024 -> 768 -> 768
-> frozen native Jittor GPT-2 small
-> causal language loss
-> Projector-only optimizer step
```

GPT-2 integrity:

```text
Model: openai-community/gpt2
Architecture: GPT2LMHeadModel
Language parameters: 124,439,808
Exported tensors: 148
Tied LM head: true
Language all stop-grad: true
Language unchanged after update: true
Validation implementation commit: c1b824490fd75a5ffbc13f3a613b56a99c12d6ca
```

Artifact hashes:

| Artifact | SHA256 |
|---|---|
| `gpt2_float32_weights.npz` | `e6da2a75ba1c4f47274e9f635a8e3cee19efd74a30009a4f4220509bbe431124` |
| `hf_config.json` | `02fad0c9969afcce718dea8ac044a4fce769c91a071d5571c124f3ae9a546c66` |
| tokenizer tree | `bdc7acfc646e49d06bf6d8635d1e06b0cb327345e7998a60e1dc927db545149c` |
| `text_reference.npz` | `699c1e2f08f061c2be933c40b4316df362a27a798341ab08b0143767b0885d13` |

Text-only logit alignment for prompt `A native Jittor language model`:

```text
shape: [1,7,50257]
max_abs_error: 0.000213623046875
atol: 0.0005
rtol: 0.0005
allclose: true
```

Five targeted native Jittor GPT-2 tests passed:

```text
Ran 5 tests in 0.155s
OK
```

Three-budget results:

| Budget | VisionZip | Projector | Packed | Loss | Compressed max abs | Prefill mean ms | Peak GPU MiB | Result |
|---:|---|---|---|---:|---:|---:|---:|---|
| 64 | `[3,65,1024]` | `[3,65,768]` | `[3,78,768]` | 9.202743 | 5.72205e-06 | 13.4518 | 1550 | PASS |
| 128 | `[3,129,1024]` | `[3,129,768]` | `[3,142,768]` | 8.336858 | 1.90735e-06 | 8.6873 | 1896 | PASS |
| 192 | `[3,193,1024]` | `[3,193,768]` | `[3,206,768]` | 8.949780 | 1.90735e-06 | 10.3456 | 2350 | PASS |

For every budget:

```text
compressed_allclose: true
assignments_exact: true
optimizer_scope_exact: true
projector_all_trainable: true
gradient_finite_and_nonzero: true
projector_changed: true
language_all_stop_grad: true
language_unchanged: true
budget passed: true
```

Top-level report:

```text
artifact_type: phase3b_native_jittor_real_gpt2_smoke_v1
real_llm: true
passed: true
```

Evidence archive transferred and verified on Windows:

```text
File: VisionZip-Jittor-phase3b-evidence-20260803.tar.gz
SHA256: 27C24946FF9AAF361A0D6A16C64F66F9FD587A8537E086BADB3DC15C0819859E
```

Timing boundary: prefill values are smoke observations, not synchronized production benchmarks. Single backward times were highly non-monotonic because of Jittor lazy execution, compilation, and cache placement. Do not use them as stable performance conclusions.

Generation boundary: decoded strings prove real GPT-2 execution and tokenizer decoding only. The Projector was randomly initialized and received one optimizer step. They do not prove caption or VQA quality.

Detailed document: `docs/PHASE3B_REAL_GPT2.md`.

## 7. Important source map

```text
visionzip_jittor/core.py
  Native VisionZip compression and PyTorch-compatible CUDA norm path.

visionzip_jittor/projector.py
visionzip_jittor/projector_config.py
  Native trainable multimodal Projector.

visionzip_jittor/multimodal.py
  Phase 3A packing and frozen-language-stub integration.

visionzip_jittor/gpt2.py
visionzip_jittor/gpt2_config.py
  Native Jittor GPT-2 blocks, tied LM head, artifact loading, loss/generation support.

reference/pytorch_visionzip.py
reference/clip_features.py
  PyTorch reference compression and real CLIP feature helpers.

scripts/export_real_clip_reference.py
scripts/run_real_clip_pipeline.py
scripts/run_jittor_alignment.py
  Phase 2 export/alignment pipeline.

scripts/run_phase3_projector_smoke.py
  Phase 3A stub integration smoke runner.

scripts/export_gpt2_jittor_artifacts.py
scripts/run_phase3b_gpt2.py
  Phase 3B artifact exporter and native real-GPT-2 runner.

configs/visionzip_64.json
configs/visionzip_128.json
configs/visionzip_192.json
configs/phase3_projector_smoke.json
configs/phase3b_gpt2.json
  Reproducible configurations.

tests/
  Config, PyTorch reference, Jittor core, Projector, GPT-2, CLIP helper, and visualization tests.
```

## 8. Generated artifacts currently available on AutoDL

Phase 2 real CLIP references:

```text
outputs/real_clip/reference_clip_64_code_exact_float32_real_clip.npz   ~64 MiB
outputs/real_clip/reference_clip_128_code_exact_float32_real_clip.npz  ~65 MiB
outputs/real_clip/reference_clip_192_code_exact_float32_real_clip.npz  ~66 MiB
outputs/real_clip/manifest.json
outputs/real_clip/visualizations/*.png  (9 files)
```

Phase 3B GPT-2 artifacts:

```text
outputs/phase3b/gpt2/gpt2_float32_weights.npz  ~475 MiB
outputs/phase3b/gpt2/hf_config.json
outputs/phase3b/gpt2/manifest.json
outputs/phase3b/gpt2/text_reference.npz
outputs/phase3b/gpt2/tokenizer/
```

Phase 3B primary reports:

```text
logs/phase3b/gpt2_smoke.json
logs/phase3b/gpt2_smoke.log
logs/phase3b/gpt2_smoke_64.json
logs/phase3b/gpt2_smoke_64.log
logs/phase3b/gpt2_unit_tests.log
logs/phase3b/export_gpt2.log
logs/phase3b/evidence/
```

The generated outputs are intentionally ignored by file-pattern rules, but their parent directories still appear as untracked because tokenizer text files and some evidence text files are not covered by the current `.gitignore`. Do not solve this by committing the artifacts. Review `.gitignore` deliberately in Phase 4A housekeeping.

## 9. Canonical verification commands

### Full unit discovery in Jittor environment

```bash
cd /root/autodl-tmp/VisionZip-Jittor
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/envs/visionzip-jittor
export OMP_NUM_THREADS=8
python -m unittest discover -s tests -v
```

Expected skips depend on which framework is installed in the active environment. Skipped PyTorch-only tests inside the Jittor environment are not automatically failures.

### Verify Phase 3B final report without rerunning the model

```bash
cd /root/autodl-tmp/VisionZip-Jittor
python - <<'PY'
import json
from pathlib import Path

path = Path("logs/phase3b/gpt2_smoke.json")
data = json.loads(path.read_text())
print("artifact_type:", data.get("artifact_type"))
print("real_llm:", data.get("real_llm"))
print("passed:", data.get("passed"))
for result in data.get("results", []):
    print(result.get("budget"), result.get("passed"))
PY
```

Expected:

```text
real_llm: True
passed: True
64 True
128 True
192 True
```

### Git safety check before a source commit

```bash
git status --short
git diff --check
git diff --cached --check
git diff --cached --stat
```

Review every staged path. Large NPZ, tokenizer artifacts, logs, and generated outputs must not enter ordinary commits.

## 10. Current limitations and non-claims

The project currently proves:

- native Jittor VisionZip behavior aligned with PyTorch synthetic inputs;
- real CLIP feature alignment for 64/128/192;
- exact discrete token/merge decisions after the CUDA numeric-path fix;
- Projector-only gradients through a frozen stub and a real frozen GPT-2;
- native Jittor GPT-2 text-logit alignment and real execution.

It does **not** yet prove:

- a trained vision-language Projector;
- useful captioning or VQA quality;
- benchmark-quality downstream metrics;
- LLaVA-equivalent behavior;
- production prefill speedup;
- stable backward latency;
- KV-cache generation;
- mixed-precision correctness;
- a larger frozen LLM integration.

## 11. Next exact actions — Phase 4A, NOT STARTED

Phase name:

```text
Phase 4A: real paired image-text Projector training infrastructure
```

Planned data path:

```text
paired image + caption
-> frozen CLIP vision features
-> native Jittor VisionZip
-> trainable Projector 1024 -> 768 -> 768
-> frozen native Jittor GPT-2
-> teacher-forced causal language loss
-> update Projector only
```

Immediate implementation order:

1. Inspect and deliberately update `.gitignore` for Phase 4 datasets, checkpoints, tokenizer artifacts, and logs without hiding source/config files.
2. Define a versioned Phase 4 training config and paired image-caption manifest schema.
3. Implement a deterministic dataset loader and train/validation split with seed recording.
4. Reuse or precompute frozen CLIP/VisionZip features to avoid unnecessary repeated vision encoding during early Projector training.
5. Implement a native Jittor Projector training runner with:
   - Projector-only optimizer scope;
   - frozen GPT-2 and frozen visual references;
   - masked causal loss;
   - finite/nonzero gradient checks;
   - periodic JSONL metrics;
   - checkpoint save/resume;
   - deterministic seed and config snapshot.
6. Add a tiny overfit/smoke dataset first. Verify loss decreases and checkpoint resume reproduces the next step before starting a larger dataset.
7. Add validation loss and deterministic generation output. Do not call generation quality successful until a proper dataset and metrics are used.
8. Add tests for dataset parsing, packing/label masks, optimizer scope, checkpoint round-trip, and a tiny training step.
9. Run Phase 4A on AutoDL inside `tmux`, archive evidence, hash the archive, update README/Phase 4 documentation, and update this file.

Decisions still open and requiring explicit documentation before a large run:

- paired dataset and license;
- dataset download/storage location;
- whether CLIP features are precomputed or encoded online;
- primary token budget for training and whether to train/evaluate all three budgets;
- maximum caption length and prompt template;
- batch size, gradient accumulation, number of epochs/steps;
- validation metrics;
- checkpoint retention policy.

A new agent must not invent completed Phase 4 results. As of this update, no Phase 4 training script, dataset, checkpoint, or quality result exists.

## 12. Network and recovery notes

### GitHub failure modes already observed

```text
curl 16 Error in the HTTP2 framing layer
GnuTLS recv error (-110)
Empty reply from server
connection timeout on port 443
```

Mitigations previously used:

```bash
git config --local http.version HTTP/1.1
git config --local http.lowSpeedLimit 0
git config --local http.lowSpeedTime 999999
source /etc/network_turbo
```

If GitHub still fails, use a Git Bundle transferred through Jupyter/Windows. This has already been used successfully and preserves exact commit history.

### Hugging Face failure modes already observed

- direct connection timeout;
- missing local cache entry;
- Xet CAS `401 Unauthorized`.

Use the environment settings in section 4.3. Do not repeatedly delete the working Hugging Face cache unless corruption is proven.

## 13. Handoff update log

| Date | Git/phase state | What changed | Next action |
|---|---|---|---|
| 2026-08-03 | `c2ac559`, Phase 3B complete | Passwordless SSH and VS Code Remote-SSH validated; created this authoritative cross-account handoff before Phase 4A. | Commit/push this file, then design and implement Phase 4A training infrastructure. |

When adding a new row, keep older rows. The newest row should state the exact commit or dirty-worktree state, the verified result, and the next blocking action.