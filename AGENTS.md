# VisionZip-Jittor Codex Handoff and Project State

> **Purpose:** This is the authoritative cross-account handoff file for Codex agents working on this reproduction. A new agent may have no access to earlier chats. Read this file completely before modifying code, running expensive jobs, changing claims, or proposing the next phase.
>
> **Last authoritative update:** 2026-08-03 (Asia/Shanghai), after the first Phase 4B materialization attempt exposed that CommonCatalog row `sha256` is distinct from the processed embedded JPEG digest. The failed assumption was corrected locally, dual-hash provenance tests pass, and no images were materialized by the failed attempt.
>
> **Current phase boundary:** Phases 1, 2, 3A, 3B, and Phase 4A are complete. **Phase 4B is in progress**. The five pinned Parquet sources are cached on AutoDL, but dataset materialization has not completed. A real-data integration defect was found and fixed locally; the next blocking action is to commit/synchronize the fix and rerun materialization in `tmux`. No Phase 4B training or quality result is claimed.

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

Last remote resource verification was on 2026-08-03 after the Phase 4A fresh and explicit-resume runs. The Phase 4B change set described below currently exists in the Windows working copy and must be committed/pushed before AutoDL preflight:

```text
Remote host: autodl-container-10894aa74d-da4e9cbe
Repository: /root/autodl-tmp/VisionZip-Jittor
Branch: main
Phase 4A implementation commit: 7a62be2 feat: add phase-four paired projector training
Synchronized Phase 4A result commit: 19029cf docs: record phase-four paired training results
GPU: NVIDIA GeForce RTX 4090, 24564 MiB
Driver: 580.105.08
nvidia-smi maximum CUDA compatibility: 13.0
Jittor nvcc toolkit: 11.8.89
Remote RAM: approximately 1.0 TiB
Phase 4A fresh run: passed=true, steps 0 -> 30
Phase 4A explicit resume: passed=true, steps 10 -> 30
```

Phase 4B synchronized source/preflight state:

```text
Infrastructure commit: 8ac45e3 (Windows, GitHub main, and AutoDL synchronized)
Dataset: common-canvas/commoncatalog-cc-by
Pinned dataset revision: 80f50fe4a1ca937f37a11be3f8eee5199d776ff3
Pilot source: five pinned 512-768 / square-aspect Parquet shards
Source rows/bytes: 9,621 / 1,263,965,106
Target samples: 8,192 (7,168 train + 1,024 validation, seed 2026)
Primary VisionZip budget: nominal 64, actual 65 including CLS
Prepared-data and feature-shard infrastructure: committed and synchronized
Windows full discovery: 50 tests OK, 13 Jittor-only skips
AutoDL full discovery: 50 tests OK, 8 PyTorch-only skips
AutoDL preparation dependencies: pyarrow 19.0.1, Pillow 10.3.0, huggingface_hub 0.36.2
AutoDL no-download preflight: passed
AutoDL estimated_required_bytes: 9,003,935,588
AutoDL free_bytes at preflight: 210,363,502,592
AutoDL Hugging Face cache at preflight: 2.2 GiB
Pinned source Parquet cache after first execute attempt: complete, about 3.3 GiB total HF cache
First materialization attempt on ba8d89e: failed safely before writing images
Observed rejection: 9,576 rows under the invalid source/embedded SHA equality assumption
Root cause: row sha256 is upstream source identity; embedded Parquet JPEG has its own digest
Local correction: preserve and validate both digests independently; 50 Windows tests pass
External dataset materialization: not completed
Feature precompute: not run
Training/evaluation: not implemented or run
```

The Phase 4A source/config/test implementation remains committed as `7a62be2`; synchronized Phase 4A handoff baseline is `212b81e`. Generated remote paths include:

```text
logs/benchmark_gpu_info.txt
logs/phase3b/
outputs/phase3b/
logs/phase4a/
logs/phase4a_resume_test/
logs/phase4a_resume_console.log
outputs/phase4a/
```

These generated paths are execution evidence, not pending source. In particular, the GPT-2 weight export is about 475 MiB and the Phase 4A checkpoints are about 15 MiB each. Never add them to ordinary Git history. Use explicit path-by-path staging; never use `git add .` or `git add -A`.

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

### 6.5 Phase 4A -- real paired Projector training infrastructure: COMPLETE

Milestone implementation commit:

```text
7a62be2 feat: add phase-four paired projector training
```

Purpose:

- turn the single-update Phase 3B path into repeated paired image-text training;
- preserve frozen precomputed CLIP/VisionZip features and frozen native GPT-2;
- train the `1024 -> 768 -> 768` Projector only;
- add deterministic data handling, target-only causal labels, JSONL metrics, complete optimizer checkpoints, and real resume;
- validate the infrastructure with a tiny deterministic overfit/smoke fixture before selecting a larger licensed dataset.

Validated data path:

```text
precomputed Phase 2 real CLIP/VisionZip tokens at nominal budget 64
-> native Jittor Projector 1024 -> 768 -> 768
-> frozen native Jittor GPT-2 small
-> teacher-forced target-only causal loss
-> Projector-only Adam update
-> JSONL metrics + complete Projector/Adam checkpoint
```

Checked-in Phase 4A inputs:

```text
configs/phase4a_tiny_overfit.json
manifests/phase4a_tiny_pairs.json
```

The manifest contains the three deterministic project-generated sample images. The deterministic split uses `dense` and `scene` for training and `text` for validation. This is an infrastructure smoke/overfit fixture, not a quality dataset.

Fresh AutoDL CUDA run:

| Field | Result |
|---|---:|
| top-level status | `passed: true` |
| start/final step | `0 / 30` |
| initial full-train loss | `9.825726509094238` |
| final full-train loss | `4.479739189147949` |
| train-loss improvement | `5.345987319946289` |
| initial validation loss | `7.5414838790893555` |
| final validation loss | `7.752180099487305` |
| Projector-only optimizer scope | `true` |
| all updates finite/nonzero | `true` |
| GPT-2 unchanged | `true` |

Checkpoint next-step CUDA replay:

```text
loss_abs_error:          4.76837158203125e-07
projector_max_abs_error: 3.725290298461914e-09
tolerance:               1e-05
bit_exact:               false
numerically_reproduced:  true
passed:                  true
```

The checkpoint loader restores the serialized Projector parameters and Adam first/second moments with exact key/shape/hash validation. The subsequent CUDA computation is numerically reproduced within `1e-5`, but is not claimed bitwise deterministic.

Explicit resume validation loaded `projector_step_000010.npz`, continued from step 10 to step 30 in separate output/log directories, and also produced `passed: true`. Its final train loss was `4.253312587738037`; all updates were finite and GPT-2 remained unchanged.

Test results:

```text
Windows: 41 tests, OK, skipped=13 (Jittor unavailable)
AutoDL/Jittor: 41 tests, OK, skipped=8 (PyTorch unavailable)
```

Important hashes:

```text
final checkpoint:
  6b3e200aa4d320b1251fced1b4a6d8fad6fad3b205ea94812a743b83c709b66b
fresh summary:
  eaf2eb9210c8ed768bfc0a4e4cbf3cf576430bedfd332f394ef417d783c7f81e
fresh metrics:
  e2907f82015e8d6d9784a9212ac69e43e68c493d39866bc9dcbfd7ce7031d76f
evidence archive (12 entries, transferred to Windows):
  01942f1fd7e82faf6eb5e8bcb9ffca9c2474b50718eea47e00ba446960926858
```

The retained validation generation was poor (`" the shows..."`). It proves execution only. Validation loss did not improve. Do not describe Phase 4A as caption-quality, generalization, or benchmark success.

Detailed document: `docs/PHASE4A_PAIRED_TRAINING.md`.

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

visionzip_jittor/phase4_config.py
  Strict Phase 4A training config, serialization, budget and resume tolerance.

visionzip_jittor/phase4_data.py
  Paired manifest parser, deterministic split/batches, Phase 2 row mapping, target masks.

visionzip_jittor/phase4_training.py
  Multimodal training batches, parameter/gradient statistics, complete atomic checkpoints, generation helpers.

scripts/run_phase4a_training.py
  End-to-end Phase 4A verifier/trainer/resumer and summary writer.

configs/phase4a_tiny_overfit.json
manifests/phase4a_tiny_pairs.json
  Reproducible tiny paired-training fixture.

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

tests/test_phase4_config_data.py
tests/test_phase4_jittor.py
  Phase 4A schema/mask/determinism, checkpoint round-trip, and tiny Projector-only update tests.

visionzip_jittor/phase4b_config.py
visionzip_jittor/phase4b_data.py
visionzip_jittor/phase4b_features.py
  Strict licensed-data plan, attribution/prepared-manifest validation, and hashed sharded feature storage.

scripts/prepare_phase4b_dataset.py
scripts/precompute_phase4b_features.py
  Default-safe preflight/materialization and frozen real CLIP/VisionZip feature precompute.

configs/phase4b_commoncatalog_cc_by_8k.json
docs/PHASE4B_DATASET_PLAN.md
tests/test_phase4b_config_data.py
  Pinned Phase 4B pilot, licensing/evaluation plan, and config/data/feature unit coverage.

tests/
  All earlier config, PyTorch reference, Jittor core, Projector, GPT-2, CLIP helper, and visualization tests.
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

Phase 4A fresh-run evidence:

```text
logs/phase4a/full_console.log
logs/phase4a/phase4a_summary.json
logs/phase4a/train_metrics.jsonl
outputs/phase4a/tiny_overfit/checkpoints/projector_step_000010.npz
outputs/phase4a/tiny_overfit/checkpoints/projector_step_000020.npz
outputs/phase4a/tiny_overfit/checkpoints/projector_step_000030.npz
```

Phase 4A explicit-resume evidence:

```text
logs/phase4a_resume_console.log
logs/phase4a_resume_test/phase4a_summary.json
logs/phase4a_resume_test/train_metrics.jsonl
outputs/phase4a/resume_test/
```

All generated models, NPZ references, checkpoints, tokenizer files, summaries, JSONL metrics, and console logs must stay out of ordinary Git commits. Evidence archives may be transferred to Windows and hashed, but should not be committed unless a future explicit repository policy says otherwise.

## 9. Canonical verification commands

### Full local checks on Windows

```powershell
cd C:\Users\69444\Desktop\cmm\VisionZip-Jittor
python -m compileall visionzip_jittor scripts tests
python -m unittest discover -s tests -v
git diff --check
```

Expected Phase 4A-aware result: 41 tests pass with 13 Jittor-dependent skips.

### Full unit discovery in the AutoDL Jittor environment

```bash
cd /root/autodl-tmp/VisionZip-Jittor
export USE_TORCH=0 USE_TF=0 USE_FLAX=0 OMP_NUM_THREADS=8
/root/autodl-tmp/envs/visionzip-jittor/bin/python   -m unittest discover -s tests -v
```

Expected Phase 4A-aware result: 41 tests pass with 8 PyTorch-dependent skips.

### Fresh Phase 4A run

```bash
cd /root/autodl-tmp/VisionZip-Jittor
rm -rf outputs/phase4a/tiny_overfit logs/phase4a
mkdir -p logs/phase4a
export USE_TORCH=0 USE_TF=0 USE_FLAX=0 OMP_NUM_THREADS=8
set -o pipefail
/root/autodl-tmp/envs/visionzip-jittor/bin/python   scripts/run_phase4a_training.py   --device cuda   2>&1 | tee logs/phase4a/full_console.log
```

### Explicit resume from step 10

```bash
/root/autodl-tmp/envs/visionzip-jittor/bin/python   scripts/run_phase4a_training.py   --device cuda   --resume outputs/phase4a/tiny_overfit/checkpoints/projector_step_000010.npz   --output-dir outputs/phase4a/resume_test   --log-dir logs/phase4a_resume_test
```

### Verify Phase 4A reports without rerunning the model

```bash
python - <<'PY'
import json
from pathlib import Path
for path in [
    Path('logs/phase4a/phase4a_summary.json'),
    Path('logs/phase4a_resume_test/phase4a_summary.json'),
]:
    data = json.loads(path.read_text())
    print(path)
    print('passed:', data.get('passed'))
    print('start_step:', data.get('start_step'))
    print('final_step:', data.get('final_step'))
    print('language_unchanged:', data.get('language_unchanged'))
    print('checkpoint_replay:', data.get('checkpoint_replay', {}).get('passed'))
PY
```

### Git safety check before a source commit

```bash
git status --short
git diff --check
git diff --cached --check
git diff --cached --stat
```

Review every staged path. Never use `git add .` or `git add -A`. Large NPZ files, tokenizer artifacts, logs, checkpoints, and generated outputs must not enter ordinary commits.

## 10. Current limitations and non-claims

The project currently proves:

- native Jittor VisionZip behavior aligned with the PyTorch reference;
- real CLIP feature alignment for nominal budgets 64/128/192;
- exact discrete token/merge decisions after the CUDA numeric-path fix;
- native Jittor GPT-2 text-logit alignment and real execution;
- Projector-only gradients through a frozen real GPT-2;
- deterministic paired-fixture parsing, splitting, batching, and target-only labels;
- repeated Projector-only training with finite updates;
- complete Projector/Adam checkpoint serialization and validated resume;
- numerical next-step replay within the declared `1e-5` CUDA tolerance.

It does **not** yet prove:

- useful image captioning or VQA quality;
- generalization beyond the three generated fixture images;
- benchmark-quality downstream metrics;
- LLaVA-equivalent behavior;
- bitwise deterministic CUDA resume;
- production prefill or training speedup;
- stable backward latency;
- KV-cache generation;
- mixed-precision correctness;
- a larger frozen LLM integration.

The fresh Phase 4A validation loss increased from `7.54148` to `7.75218`, and the generated validation text was poor. Retain both facts in future reports. They do not invalidate the infrastructure acceptance result, but they prohibit any visual-language quality claim.

## 11. Next exact actions -- Phase 4B, IN PROGRESS

Phase name:

```text
Phase 4B: licensed paired-dataset training and held-out caption evaluation
```

The dataset decision and first infrastructure slice are fixed:

```text
Dataset: common-canvas/commoncatalog-cc-by
Revision: 80f50fe4a1ca937f37a11be3f8eee5199d776ff3
Pilot: 8,192 accepted samples from five pinned Parquet objects
Split: 7,168 train / 1,024 validation, seed 2026
Caption: blip2_caption, normalized and length-filtered
License: row-level Creative Commons Attribution URLs only
Vision encoder: openai/clip-vit-large-patch14-336 at pinned revision
VisionZip: nominal budget 64, 65 output tokens including CLS
Feature store: 32 expected NPZ shards of 256 samples, float32
```

Exact next actions:

1. Commit/push the dual-hash correction and documentation/tests; fast-forward AutoDL without staging generated logs or cached data.
2. Rerun `scripts/prepare_phase4b_dataset.py --execute` in `tmux`, reusing the pinned cached Parquet sources, then verify exact counts, both digest fields, attribution fields, decoded dimensions, and the completed manifest.
3. In `tmux`, precompute and verify all frozen feature shards; test `--verify-existing` and interrupted-shard reuse.
4. Implement the next source slice: gradient accumulation, rolling/best/final checkpoint retention, held-out NLL/perplexity, BLEU-1/BLEU-4, ROUGE-L, and deterministic generation subset.
5. Run a fresh CUDA training/evaluation and an explicit resume run, archive/hash evidence, update README/this handoff, and only then call Phase 4B complete.

Current claim boundary: Phase 4B has started at the planning and infrastructure level. No real paired-data materialization, feature-precompute, training, held-out improvement, or caption-quality claim exists yet.

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
| 2026-08-03 | `7a62be2` plus the documentation commit containing this row | Implemented paired-manifest training, Projector-only multi-step optimization, complete Projector/Adam checkpoints and resume; Windows and AutoDL tests passed; fresh and explicit-resume CUDA runs both recorded `passed: true`. | Evidence archive created, transferred, and SHA256-verified; push the Phase 4A commits and fast-forward AutoDL, then plan Phase 4B. |
| 2026-08-03 | `19029cf` synchronized on Windows, GitHub, and AutoDL | Phase 4A source/docs are committed and pushed; both environments point to the same commit; the 12-entry evidence archive is present on Windows with verified SHA256 `01942F...6858`. | Begin Phase 4B only by first writing the licensed-dataset plan; do not claim external-dataset training has started. |
| 2026-08-03 | `212b81e` baseline plus local Phase 4B infrastructure change set | Selected and pinned CommonCatalog CC-BY, fixed the 8,192-sample pilot and held-out policy, implemented safe preparation/preflight and hashed feature-shard modules/scripts, passed 9 focused tests and 50-test full Windows discovery, and passed a Windows no-download preflight. No download or training has run. | Commit/push, sync AutoDL, then run import smoke and the AutoDL no-download preflight. |
| 2026-08-03 | `8ac45e3` synchronized on Windows, GitHub, and AutoDL; handoff update in the following documentation commit | Installed only the pinned Phase 4B preparation dependency set required on AutoDL, passed all 50 AutoDL tests with 8 PyTorch-only skips, and passed the AutoDL no-download preflight with 210,363,502,592 free bytes versus 9,003,935,588 estimated required bytes. No external data was downloaded. | Launch the pinned 8,192-sample materialization in `tmux`, then validate the completed dataset manifest before feature precompute. |
| 2026-08-03 | `ba8d89e` synchronized plus local dual-hash correction | Installed `tmux`, cached all five pinned Parquet sources, and ran the first execute attempt. It failed safely with 0 images because 9,576 otherwise acceptable rows had distinct row-source and embedded-JPEG SHA256 values. Real-row inspection established the two provenance digests are intentionally distinct; code/docs/tests now preserve both, reject malformed source digests, and all 50 Windows tests pass. | Commit/push/sync the correction, then rerun materialization from the completed source cache in `tmux`. |

When adding a new row, keep older rows. The newest row should state the exact commit or dirty-worktree state, the verified result, and the next blocking action.
