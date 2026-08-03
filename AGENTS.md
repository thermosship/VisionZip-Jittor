# VisionZip-Jittor Codex Handoff and Project State

> **Purpose:** This is the authoritative cross-account handoff file for Codex agents working on this reproduction. A new agent may have no access to earlier chats. Read this file completely before modifying code, running expensive jobs, changing claims, or proposing the next phase.
>
> **Last authoritative update:** 2026-08-03 (Asia/Shanghai), after completing the final clean-checkout README walkthrough at commit `04f098d8d69edd0ad85da37351892a3685ac475b`. The walkthrough used an independent GitHub checkout and a fresh conda prefix, installed from checked-in requirements, and passed the activation contract, 11-script help smoke, 80-test discovery (`8` environment-only skips), compileall, diff checks, synthetic alignment, real CLIP 64/128/192 alignment, native Jittor real GPT-2 smoke, and Phase 4B no-download preflight. The machine-readable result is `docs/results/clean_readme_walkthrough_04f098d.json`; external walkthrough logs remain untracked on AutoDL and the compact summary SHA256 is `5b8ebf154a72d6c3951910330fcdadc250611fcaa1461c30fb5b70fa6cc03f01`. Walkthrough documentation also passed Windows review: the focused environment-script suite passed 3/3, full discovery passed 80 tests with 18 environment-only skips, compileall and diff checks passed, 50 Markdown relative links resolved, and the compact JSON integrity check passed. The reviewed walkthrough record was committed as `9671d36 docs: record clean README walkthrough`, pushed to GitHub, and fast-forwarded on AutoDL without touching untracked artifacts. The immutable submission tag/release freeze is now the only blocking repository action. Phase 5B remains deliberately deferred.
>
> **Current phase boundary:** **Phases 1, 2, 3A, 3B, 4A, 4B, and 5A are complete. The clean README walkthrough is complete and Submission Readiness release freeze is active. Phase 5B is deferred.** The next blocking work is to commit/push this final release-state handoff, freeze the annotated submission tag and GitHub Release, then move to PPT/video production. Do not describe this project as a full reproduction of every VisionZip paper experiment, LLaVA-equivalent training, caption-quality improvement, raw-logit bitwise equality, universal strict-`1e-5` equality, or universal speedup.

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

Windows, GitHub, and AutoDL are synchronized at `9671d36 docs: record clean README walkthrough`. The clean walkthrough and its reviewed documentation are committed. This final handoff update is the only tracked release-state change before tagging. Existing AutoDL model/data/checkpoint/log artifacts remain untracked and must not be added to Git. Historical Phase 2/4A/4B/5A evidence archives remain authoritative for the measured values used by compact tables and plots.

Current Submission Readiness state:

```text
Walkthrough target commit: 04f098d8d69edd0ad85da37351892a3685ac475b
Fresh checkout: /root/autodl-tmp/submission_walkthrough/VisionZip-Jittor-04f098d
Fresh conda prefix: /root/autodl-tmp/envs/visionzip-readme-clean-04f098d
Final walkthrough status: passed=true
Tests: Ran 80 tests in 23.385s; OK (skipped=8)
Passed gates: activation contract, 11-script help smoke, compileall, git diff --check, synthetic alignment, real CLIP 64/128/192, real GPT-2 smoke, Phase 4B no-download preflight
External log directory: /root/autodl-tmp/VisionZip-Jittor/logs/submission_readiness/clean_walkthrough_04f098d/
Compact result: docs/results/clean_readme_walkthrough_04f098d.json
External compact-summary SHA256: 5b8ebf154a72d6c3951910330fcdadc250611fcaa1461c30fb5b70fa6cc03f01
Walkthrough record commit: 9671d36 docs: record clean README walkthrough
Synchronization: Windows, GitHub main, and AutoDL main all at 9671d36; AutoDL untracked evidence preserved
Release policy: source and release notes only; no weights, datasets, checkpoints, feature shards, CLIP references, or large raw logs
Windows validation: focused 3/3; full 80 tests with 18 skips; compileall/diff checks pass; 50 relative links and compact JSON pass
Next blocker: commit/push this final handoff state, then annotated tag and GitHub Release
```

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
Dual-hash correction commit: 14e527f (Windows, GitHub main, and AutoDL synchronized)
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
Correction: preserve and validate both digests independently; 50 Windows tests pass
Second materialization attempt on 14e527f: completed successfully, exit code 0
Prepared dataset directory: datasets/phase4b/commoncatalog_cc_by_8k (about 1.1 GiB)
Materialized images/samples: 8,192 / 8,192
Materialized split: 7,168 train + 1,024 validation
Full dataset validation: passed=true
Validated samples SHA256: c2b205622a67349ee22547e2648c264cd6a71979906d57cc4b59fede10e330f2
Pinned source files: 5; recomputed source SHA256 and byte sizes all exact
Distinct source/embedded hashes: 8,192; invalid attribution and dimension mismatches: 0
First feature-precompute launch on 4d376c8: failed before writing any shard
Failure: resolve_layer_index received requested=-2 as num_layers and raised num_layers must be positive
Correction commit: 42534b4; explicit num_layers/requested keyword call
Windows full discovery after correction: 50 tests OK, 13 Jittor-only skips
Second feature-precompute launch on 42534b4: completed, exit code 0
Feature precompute result: passed=true; PyTorch 2.1.2+cu118; Transformers 4.31.0; CUDA
Feature manifest SHA256: 2673aafd3ec7084c7eae54cd8eaac693fc21f84892cccf60a0c14f8c349a36a9
Feature config SHA256: ff80fda5bfc9a56580ccd26ede7cf4f3a8ea3742c10c4d6f55dc5afa8dbbe6ac
Feature shards: 32 x 256 samples; 8,192 total; 2,027,769,839 shard bytes
Compressed tokens: float32 [N, 65, 1024], all finite
Feature sample order: exact match to prepared dataset; all IDs unique
Selected indices and assignments: integer shapes/ranges valid; selected indices unique per row
Pinned model/revision/layer/config/dataset hashes: all exact
--verify-existing: passed=true, reused=true, exit code 0
Reuse safety: valid shard accepted; truncated NPZ rejected with BadZipFile
Feature validation: passed=true
Synchronized frozen-feature documentation commit: 3c8a3f1
Trainer/evaluator source commit: 1eadddd feat: add phase-four licensed training runner
AutoDL source/test synchronization: HEAD 1eadddd; all 56 tests OK with 8 environment-only skips
Phase 4B trainer/evaluator files: visionzip_jittor/phase4b_training.py, scripts/run_phase4b_training.py, tests/test_phase4b_training.py
Checkpoint compatibility: Phase 4A default artifact remains unchanged; Phase 4B uses phase4b_projector_checkpoint_v1
Training schedule: deterministic resume-safe shuffled stream, microbatch 4 x accumulation 4, target-token-weighted effective-batch NLL
Optimizer correction: reset Jittor Adam n_step to completed_optimizer_steps + 1 immediately before each accumulated update
Scheduler: 67-step linear warmup then cosine decay; final configured update remains positive
Evaluation: exact held-out target NLL/perplexity plus deterministic 128-sample BLEU-1, add-one-smoothed BLEU-4, and mean ROUGE-L F1
Checkpointing: rolling last 4, separate best checkpoint, final checkpoint, optimizer-step-boundary resume identity checks
First CUDA fresh smoke directory: outputs/phase4b/commoncatalog_cc_by_8k/smoke_fresh
First CUDA fresh smoke result: two valid optimizer updates, final_optimizer_step=2, optimizer_n_step=2, all_updates_finite=true
First CUDA fresh smoke integrity: language_unchanged=true, projector_optimizer_scope_exact=true, projector_max_parameter_delta=4.482455551624298e-06
First CUDA fresh smoke held-out NLL: 6.643716576688785 at step 0; 6.614728381051471 at step 2
First CUDA fresh smoke checkpoint: checkpoints/projector_step_000002.npz
First CUDA fresh smoke final status: exit code 1, passed=false, projector_all_trainable=false
Diagnosed root cause: Jittor 1.3.11 projector.eval() marks all Projector parameters stop-grad; projector.train() restores them without changing parameter values
Eval-state correction commit: 92cb004 fix: restore phase-four projector state after eval
Correction behavior: restore projector.train() after final evaluation/generation before checking trainability; record projector_trainability_restored_after_evaluation
Windows tests after correction: 57 tests OK with 15 environment-only skips
AutoDL tests after correction: 57 tests OK with 8 environment-only skips; focused eval/train restoration test passed
Corrected fresh smoke directory: outputs/phase4b/commoncatalog_cc_by_8k/smoke_fixed_92cb004
Corrected fresh smoke result: exit code 0, passed=true, completed_training=false, steps 0 -> 2, optimizer_n_step=2
Corrected fresh smoke invariants: language_unchanged=true, projector_optimizer_scope_exact=true, projector_all_trainable=true, all_updates_finite=true
Corrected fresh smoke held-out NLL: 6.643716430983611 at step 0; 6.614728552631185 at step 2
Corrected explicit resume directory: outputs/phase4b/commoncatalog_cc_by_8k/smoke_resume_92cb004
Corrected explicit resume result: exit code 0, passed=true, completed_training=false, steps 2 -> 4, optimizer_n_step=4
Corrected resume invariants: language_unchanged=true, projector_optimizer_scope_exact=true, projector_all_trainable=true, all_updates_finite=true
Corrected resume held-out NLL: 6.6147287388124845 at loaded step 2; 6.532555884020771 at step 4
The tiny difference between the fresh step-2 NLL and resumed initial step-2 NLL is within the declared CUDA numerical tolerance and does not alter discrete schedule/checkpoint acceptance
Full run source commit: 88e6e8ba2823758aee724edb811eb0824a297380
Full run directory: outputs/phase4b/commoncatalog_cc_by_8k/training_88e6e8b
Full log directory: logs/phase4b/training_88e6e8b
Full run result: exit code 0, passed=true, completed_training=true, steps 0 -> 1344, optimizer_n_step=1344
Full run invariants: finite updates, unchanged/frozen 124,439,808-parameter GPT-2, exact Projector-only optimizer scope, restored trainable Projector
Held-out validation: 1,024 samples / 10,744 target tokens
Held-out NLL: 6.643716854607091 -> 2.4616965070888344 (62.94699848050014% reduction)
Held-out perplexity: 767.9440313203165 -> 11.724685689282781 (65.49804844852032x lower)
Best/final step: 1344; best and final checkpoint SHA256 f9c6ea2d970938f03191d0027aa444f9affc99f899a1d26ab64b3274c4755c15
Deterministic 128-sample generation: BLEU-1 0.31536426498207987, BLEU-4 0.07713203296811436, ROUGE-L 0.2976671870474831
Generation boundary: one BLIP-2 synthetic reference per image; not directly comparable with multi-reference COCO metrics; visibly poor examples remain
First evidence archive on Windows: VisionZip-Jittor-phase4b-evidence-20260803.tar.gz
First evidence archive SHA256: 239AA45FDE838E2820AAC3D8B3546B5A9AD827FCF5DD471499D4F4240A47E88A
Remaining acceptance gap: the successful summary lacks runner-measured post-warm-up throughput and peak process GPU memory
Benchmark correction commit: 20c8b59 feat: record phase-four training benchmark evidence; old run/archive are preserved
Benchmark handoff commit: 0f53a93 docs: record phase-four benchmark evidence gap
Benchmark smoke handoff commit: 8b51001 docs: record phase-four benchmark smoke
First benchmark-aware AutoDL full discovery: intermittent Jittor process crash preserved in logs/phase4b/benchmark_0f53a93/autodl_tests.log
Affected isolated test: passed
AutoDL full discovery retry 1: 60 tests OK with 8 environment-only skips
Benchmark smoke directory: outputs/phase4b/commoncatalog_cc_by_8k/smoke_benchmark_0f53a93
Benchmark smoke result: exit code 0, passed=true, steps 0 -> 68, benchmark accepted
Benchmark smoke post-warm-up step 68: 117.05001987412734 effective samples/s, 1119.2908150463427 target tokens/s, 1562 MiB current-process peak GPU memory
Final benchmark run source: 8b510017e4250bc626ca001bcf56cc05b7e09fa9
Final benchmark run directory: outputs/phase4b/commoncatalog_cc_by_8k/training_benchmark_0f53a93
Final benchmark log directory: logs/phase4b/training_benchmark_0f53a93
Final benchmark run result: exit code 0, passed=true, completed_training=true, steps 0 -> 1344, optimizer_n_step=1344
Final benchmark invariants: all updates finite, GPT-2 all stop-grad and hash unchanged, exact Projector-only optimizer scope, Projector trainable after evaluation
Final held-out NLL: 6.643716599545368 -> 2.413187829110486 (63.6771407546851% reduction)
Final held-out perplexity: 767.9438354472138 -> 11.169510943754625 (68.7535774228867x lower)
Final 128-sample generation: BLEU-1 0.28304947283049475, BLEU-4 0.05727683512769526, ROUGE-L 0.26176086973563917
Accepted benchmark range: steps 68 -> 1344, 1277 measured updates, mean 120.37183717184918 ms/update
Accepted throughput: 132.92145717737577 effective samples/s and 1413.1492154945145 target tokens/s
Accepted current-process GPU peak: 3058 MiB from 1416 samples at 0.1-second interval
Final benchmark summary/checkpoint hashes: b13532ea76352ebe9fcb3e1c3e0a4c6018f7865f930adc21435503e6daff4b80 / d209bba83b7ee822efa5e29912ce0bc82748fdcb90303f06981f912ee9f928fa
Phase 4B completion archive: final v2 archive created and cross-host verified; the first archive is preserved
```


Phase 5A current development state:

```text
Baseline HEAD on Windows and AutoDL: 577dd56 docs: mark phase four complete
Tracked modifications:
  visionzip_jittor/gpt2.py
  tests/test_gpt2_jittor.py
Untracked source/config/docs:
  configs/phase5a_kv_cache.json
  docs/PHASE5A_KV_CACHE_PLAN.md
  scripts/run_phase5a_kv_cache.py
Do not stage generated logs or outputs.
```

Implemented cache contract:

```text
Per-layer cache: (key, value)
Cache tensor shape: [batch, heads, cached_tokens, head_dim]
GPT-2 small expected layer/head/head_dim: 12 / 12 / 64
Prefill: packed prompt + projected real visual tokens + generation suffix
Decode: one generated token at a time with past_key_values
Uncached full recomputation: retained as correctness oracle
Token acceptance: cached/uncached greedy IDs exact
Logit acceptance: every step allclose at atol=rtol=1e-5
Model invariants: GPT-2 and Phase 4B Projector stop-grad and SHA256 unchanged
Performance claim: report timings/memory; speedup > 1 is not required
Quality boundary: no caption-quality improvement claim
```

Development validation already completed on AutoDL:

```text
Focused GPT-2 test log: logs/phase5a/gpt2_tests_runner_initial.log
Focused GPT-2 result: 8 tests OK
Earlier full discovery log: logs/phase5a/autodl_tests_initial.log
Earlier full discovery result: 63 tests OK, 8 skipped
Current full discovery log: logs/phase5a/autodl_tests_runner.log
Current full discovery exit code: logs/phase5a/autodl_tests_runner.exitcode = 0
Current full discovery result: 63 tests OK, 8 skipped, 0.936 seconds
Real GPT-2 cache diagnosis: logs/phase5a/real_gpt2_cache_initial.log
Diagnosis result: max_abs_error=0.00014495849609375, allclose_1e5=true,
                  argmax_exact=true, 12 cache layers, shape [1,12,7,64]
```

Formal runner smoke history:

```text
Temporary smoke config: /tmp/phase5a_smoke.json
Scope: budget 64, sample index 0, max_new_tokens=2,
       warmup_runs=0, measured_runs=1
First log: logs/phase5a/kv_cache_runner_smoke.log
First result: failed only phase4b_config_hash_exact
Root cause: raw Phase 4B config file SHA256 was incorrectly compared with
            the checkpoint canonical JSON SHA256
Fix: canonical_json_sha256(phase4b_config.to_dict())
Retry log: logs/phase5a/kv_cache_runner_smoke_retry1.log
Retry exit code: logs/phase5a/kv_cache_runner_smoke_retry1.exitcode
Retry JSON: logs/phase5a/kv_cache_runner_smoke_retry1.json
Retry 1 result: passed=true, invariants_passed=true
Retry 1 audit finding: nominal cached_decode_only still performed prefill
                       inside the timed function; preserve but do not publish
Fix: precompute immutable prefill cache and first token outside decode timer
Corrected retry 2 log: logs/phase5a/kv_cache_runner_smoke_retry2.log
Corrected retry 2 exit code: logs/phase5a/kv_cache_runner_smoke_retry2.exitcode = 0
Corrected retry 2 JSON: logs/phase5a/kv_cache_runner_smoke_retry2.json
Corrected retry 2 result: passed=true, invariants_passed=true
Cached prefill mean: 4.1622 ms
Cached decode-only mean: 7.3734 ms
Cached total mean: 15.9335 ms
Uncached total mean: 18.7474 ms
Uncached/cached total speedup: 1.17660x
Peak process GPU memory: 1122 MiB
Final cache: 12 layers of [1,12,78,64]
```

The smoke result is implementation evidence only because the source tree was dirty and `source_commit` therefore identifies only baseline HEAD `577dd56`. Do not publish these timings as the formal Phase 5A benchmark. Commit and synchronize first, then rerun the full checked-in protocol from a clean tree under `tmux`.

The Phase 4A source/config/test implementation remains committed as `7a62be2`; synchronized Phase 4A handoff baseline is `212b81e`. Generated remote paths include:

```text
logs/benchmark_gpu_info.txt
logs/phase3b/
outputs/phase3b/
logs/phase4a/
logs/phase4a_resume_test/
logs/phase4a_resume_console.log
outputs/phase4a/
logs/phase5a/
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

### 6.6 Phase 4B -- licensed real paired training: COMPLETE

Validated data and frozen features:

```text
Dataset: common-canvas/commoncatalog-cc-by
Pinned revision: 80f50fe4a1ca937f37a11be3f8eee5199d776ff3
Accepted samples: 8,192 (7,168 train / 1,024 validation)
Caption/reference: blip2_caption, one synthetic reference per image
Feature store: 32 exact-order float32 shards, VisionZip nominal budget 64 / 65 tokens with CLS
Prepared samples SHA256: c2b205622a67349ee22547e2648c264cd6a71979906d57cc4b59fede10e330f2
Feature manifest SHA256: 2673aafd3ec7084c7eae54cd8eaac693fc21f84892cccf60a0c14f8c349a36a9
```

Resume and test evidence:

```text
Corrected fresh smoke: steps 0 -> 2, passed=true, optimizer_n_step=2
Corrected explicit resume: steps 2 -> 4, passed=true, optimizer_n_step=4
Benchmark smoke: steps 0 -> 68, passed=true, training_benchmark_accepted=true
Windows: 60 tests OK, 15 environment-only skips
AutoDL retry 1: 60 tests OK, 8 environment-only skips
Preserved history: one intermittent Jittor full-suite process crash and the original eval-state false negative
```

Final benchmark-instrumented complete run:

```text
Source commit: 8b510017e4250bc626ca001bcf56cc05b7e09fa9
Optimizer steps: 1,344 / 1,344
Effective batch: 4 microbatch x 4 accumulation = 16 samples/update
Result: passed=true, completed_training=true, training_benchmark_accepted=true
Initial held-out NLL/PPL: 6.643716599545368 / 767.9438354472138
Final held-out NLL/PPL: 2.413187829110486 / 11.169510943754625
NLL relative reduction: 63.6771407546851%
Perplexity reduction: 68.7535774228867x
Best/final step: 1,344
Projector max parameter delta: 0.012454323470592499
Summary SHA256: b13532ea76352ebe9fcb3e1c3e0a4c6018f7865f930adc21435503e6daff4b80
Metrics JSONL SHA256: f4ab48d472fb1cc74376e0661f3177c8f31d1f3e51eeb895f1173bdb32f6f0ca
Final checkpoint SHA256: d209bba83b7ee822efa5e29912ce0bc82748fdcb90303f06981f912ee9f928fa
```

Accepted post-warm-up benchmark:

```text
Warm-up: optimizer steps 1 -> 67
Measured optimizer steps: 68 -> 1344 (1,277 updates)
Mean optimizer-step compute: 120.37183717184918 ms
Effective throughput: 132.92145717737577 samples/s
Target-token throughput: 1413.1492154945145 tokens/s
Current-process peak GPU memory: 3058 MiB
Memory sampling: 1,416 samples at 0.1-second interval
```

Generation evaluation boundary:

```text
Samples: deterministic held-out subset of 128
BLEU-1: 0.28304947283049475
BLEU-4: 0.05727683512769526
ROUGE-L: 0.26176086973563917
Reference policy: one BLIP-2 synthetic caption per image
Non-claim: not directly comparable with multi-reference COCO metrics and does not prove high-quality captioning
```

The technical acceptance gap and administrative evidence step are both closed. Final v2 archive `VisionZip-Jittor-phase4b-evidence-final-v2-20260803.tar.gz` contains 45 files and matched SHA256 `2EFEEAA88F18AB11B8431A7DD810B296366073D14B5717D02C72152DBA70C032` on AutoDL and Windows; the original archive remains preserved.


### 6.7 Phase 5A -- native Jittor GPT-2 KV-cache generation: COMPLETE

Fixed implementation and formal protocol:

```text
Plan: docs/PHASE5A_KV_CACHE_PLAN.md
Config: configs/phase5a_kv_cache.json
Runner: scripts/run_phase5a_kv_cache.py
Acceptance-v3 source commit: 72277174069b9cef63831529f3ef2e3e16965cd1
Budgets: 64, 128, 192
Samples: Phase 2 rows 0, 1, 2 (dense.png, scene.png, text.png)
Generated tokens: 32
Warmups / measured runs: 3 / 10
Acceptance: exact greedy IDs + exact cache contract + per-step total variation <= 5e-5
Diagnostics retained: raw logits, centered logits, coordinatewise probability allclose
Speedup > 1: not required
```

Formal clean result from `7227717`:

| Budget | Exact IDs/cache/TV | Maximum TV | Cached total | Uncached total | Speedup | Peak GPU |
|---:|:---:|---:|---:|---:|---:|---:|
| 64 | 3/3 passed | `2.0911763919726617e-05` | `240.8540 ms` | `243.3399 ms` | `1.01032x` | `1168 MiB` |
| 128 | 3/3 passed | `3.505422367609121e-05` | `241.4139 ms` | `255.4643 ms` | `1.05820x` | `1250 MiB` |
| 192 | 3/3 passed | `1.598435777075381e-05` | `241.0673 ms` | `290.8731 ms` | `1.20661x` | `1322 MiB` |

All 9/9 budget/sample cases passed exact greedy-token, exact cache-contract, frozen-model/projector, and TV gates. Clean AutoDL discovery passed 74 tests with 8 environment-only skips. The formal result remains protocol- and device-specific: raw-logit strict `1e-5` equality is diagnostic rather than the acceptance gate, and the measured speedups are not a universal performance claim.

Final evidence archive:

```text
Archive: VisionZip-Jittor-phase5a-evidence-20260803.tar.gz
Checksummed entries: 112
Independent unpack verification: 112/112
Size: 521499 bytes
AutoDL/Windows SHA256: 20093fb7550d6e17fc96566191236bc3631952998a5d4097d245ae1f2037ec81
```

The archive preserves formal, failed, dirty, and diagnostic namespaces while excluding large model/data artifacts whose provenance and hashes are recorded separately.

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
  Native Jittor GPT-2 blocks, tied LM head, artifact loading, loss/generation support, and Phase 5A per-layer KV caching.


configs/phase5a_kv_cache.json
  Versioned Phase 5A correctness and repeated benchmark protocol.

docs/PHASE5A_KV_CACHE_PLAN.md
  Fixed scope, artifacts, acceptance gates, smoke status, and non-claims.

scripts/run_phase5a_kv_cache.py
  Real GPT-2 + Phase 4B Projector + Phase 2 real-CLIP cached/uncached runner.

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

visionzip_jittor/phase4b_training.py
scripts/run_phase4b_training.py
tests/test_phase4b_training.py
  Deterministic full-store loader, resume-safe batching/LR, caption metrics, rolling checkpoint policy, frozen-GPT-2 Projector training, held-out evaluation, and post-warm-up throughput/current-process peak-GPU-memory evidence aggregation. The original trainer is AutoDL-validated; benchmark instrumentation is committed locally as `20c8b59` and awaits push/AutoDL validation/rerun.

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

Current Phase 4B benchmark-aware result: 60 tests pass with 15 environment-only skips.

### Full unit discovery in the AutoDL Jittor environment

```bash
cd /root/autodl-tmp/VisionZip-Jittor
export USE_TORCH=0 USE_TF=0 USE_FLAX=0 OMP_NUM_THREADS=8
/root/autodl-tmp/envs/visionzip-jittor/bin/python   -m unittest discover -s tests -v
```

Expected after synchronizing the benchmark patch: 60 tests pass with 8 PyTorch-dependent skips.

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
- licensed 8,192-sample external paired-data materialization with row-level provenance and deterministic 7,168/1,024 split;
- 32 verified frozen CLIP/VisionZip feature shards;
- repeated Projector-only training with finite updates, frozen/hash-unchanged GPT-2, held-out NLL/perplexity, and single-synthetic-reference generation metrics;
- complete Projector/Adam checkpoint serialization and validated resume;
- numerical next-step replay within the declared `1e-5` CUDA tolerance;
- runner-measured post-warm-up training throughput, sampled process peak GPU memory, and cross-host evidence integrity.
- clean-commit native Jittor GPT-2 KV-cache generation with exact 9/9 greedy sequences, exact cache contracts, and a frozen per-step TV distribution gate.

It does **not** yet prove:

- human-level, production-quality, or state-of-the-art image captioning/VQA;
- direct comparability with multi-reference COCO metrics;
- quality scaling beyond the fixed 8,192-sample pilot and frozen GPT-2-small setup;
- LLaVA-equivalent behavior;
- bitwise deterministic CUDA resume;
- production prefill or training speedup;
- stable backward latency;
- mixed-precision correctness;
- a larger frozen LLM integration.

The fresh Phase 4A validation loss increased from `7.54148` to `7.75218`, and the generated validation text was poor. Retain both facts in future reports. They do not invalidate the infrastructure acceptance result, but they prohibit any visual-language quality claim.

## 11. Next exact actions -- SUBMISSION READINESS

Phase 5A and the clean README walkthrough are complete; Phase 5B is deferred. The blocking work is now the immutable repository release freeze and presentation materials, not additional model scope.

Exact next actions:

1. Explicitly commit/push this final release-state `AGENTS.md` update and fast-forward AutoDL without touching its untracked artifacts.
2. Verify Windows, GitHub `main`, and AutoDL are synchronized and that the Windows tracked tree is clean.
3. Confirm `v0.1.0-jittor-submission` does not already exist; create and push it as an annotated immutable tag at the final release-state commit, and never move or overwrite it.
4. Create a GitHub Release from that tag. Publish source and release notes only; do not upload model weights, datasets, checkpoints, feature shards, CLIP reference NPZ files, or raw large logs.
5. Recheck all relative links from the tagged tree and recheck the live `GrokCV/Jittor-Sprouts` list on the actual submission date.
6. Move to PPT, narration, and video. Do not start Phase 5B unless the user explicitly reprioritizes it and a versioned scope is reviewed first.

Current claim boundary: this repository demonstrates a native-Jittor VisionZip compression core, real CLIP feature alignment, frozen real GPT-2 integration, Projector-only paired training on a pinned 8,192-sample licensed subset, and native KV-cache generation under pinned protocols. It does not reproduce every experiment or quality conclusion in the VisionZip paper and does not establish LLaVA-equivalent quality, COCO multi-reference caption quality, raw-logit bitwise equality, universal strict-`1e-5` equality, or universal speedup.

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
| 2026-08-03 | `14e527f` synchronized on Windows, GitHub, and AutoDL; dataset artifacts remain untracked | Reran the corrected materialization from the pinned cache in `tmux`; exit code was 0 and all 8,192 images were written. Independent full validation passed with 7,168/1,024 train/validation samples, sample-manifest SHA256 `c2b205622a67349ee22547e2648c264cd6a71979906d57cc4b59fede10e330f2`, all five source shard hashes/sizes exact, 8,192 distinct source/embedded digest pairs, and zero attribution or decoded-dimension errors. | Commit/push this handoff update, sync AutoDL, then precompute and verify the 32 frozen CLIP/VisionZip feature shards in `tmux`. |
| 2026-08-03 | `4d376c8` synchronized plus local feature-precompute call-site correction | Launched `phase4b-features`; it failed safely before writing shards with `ValueError: num_layers must be positive`. Inspection found `scripts/precompute_phase4b_features.py` passed `(requested_layer_index, layer_count)` to the helper whose contract is `(num_layers, requested)`. The call now uses explicit keyword arguments; targeted tests and all 50 Windows tests pass. | Commit/push/sync the correction, then relaunch feature precompute in `tmux` and inspect the first completed shard before leaving the long job unattended. |
| 2026-08-03 | `42534b4` synchronized on Windows, GitHub, and AutoDL; feature artifacts remain untracked | Relaunched frozen feature precompute in `tmux`; it completed with exit code 0 and produced 32 shards covering all 8,192 samples. Independent validation passed: exact model/revision/layer/config/dataset hashes, exact sample order, float32 `[N,65,1024]` finite tokens, legal discrete arrays, manifest SHA256 `2673aafd3ec7084c7eae54cd8eaac693fc21f84892cccf60a0c14f8c349a36a9`, successful `--verify-existing` reuse, and rejection of a truncated shard. | Commit/push this handoff result, then implement the Phase 4B trainer/evaluator and its focused tests before launching real training. |
| 2026-08-03 | `3c8a3f1` synchronized plus uncommitted Windows Phase 4B trainer/evaluator slice | Implemented single-allocation exact-order feature loading, deterministic optimizer-step batches, target-token-weighted gradient accumulation with Jittor Adam `n_step` correction, warmup/cosine LR, Phase 4B checkpoint/resume identity, rolling/best/final retention, held-out NLL/perplexity, deterministic single-reference BLEU/ROUGE generation evaluation, and intentional stop-at-step support. All 56 Windows tests pass with 14 environment-only skips. No AutoDL test or training result exists yet. | Explicitly commit/push/sync source and handoff files, run AutoDL Jittor tests, then perform a fresh step-2 and resumed step-4 CUDA smoke before the 1,344-step run. |

| 2026-08-03 | `1eadddd` synchronized plus uncommitted Windows eval-state correction | AutoDL passed all 56 tests with 8 skips and completed two finite CUDA optimizer updates with exact Projector-only scope, unchanged GPT-2, `optimizer_n_step=2`, a step-2 checkpoint, and held-out NLL moving from `6.643716576688785` to `6.614728381051471`. The process then exited 1 only because Jittor 1.3.11 `projector.eval()` set Projector parameters stop-grad before the final trainability check. A Windows fix now calls `projector.train()` before the invariant, records the restoration, adds a focused Jittor test, and all 57 Windows tests pass with 15 skips. | Commit/push/sync the fix, run AutoDL tests, then rerun fresh step 2 and explicit resume step 2 -> 4 in new directories while preserving the failed smoke history. |

| 2026-08-03 | `92cb004` synchronized; corrected smoke/resume artifacts untracked | Added post-evaluation `projector.train()` restoration and a focused Jittor regression test. All 57 tests pass on Windows (15 skips) and AutoDL (8 skips). Corrected fresh smoke passed steps 0 -> 2 with `optimizer_n_step=2`, unchanged frozen GPT-2, exact optimizer scope, trainable Projector, finite updates, and NLL `6.643716430983611 -> 6.614728552631185`. Explicit resume passed steps 2 -> 4 with `optimizer_n_step=4` and NLL `6.6147287388124845 -> 6.532555884020771`. Original failed smoke history remains preserved. | Commit/sync this result record, then launch the fixed 1,344-step run in a new tmux/output/log namespace and monitor through the first step-112 evaluation/checkpoint. |

| 2026-08-03 | `20c8b59` committed locally; push/AutoDL validation pending | The first full 1,344-step CUDA run passed with unchanged frozen GPT-2 and held-out NLL `6.643716854607091 -> 2.4616965070888344`; deterministic 128-sample single-synthetic-reference generation recorded BLEU-1 `0.31536426498207987`, BLEU-4 `0.07713203296811436`, and ROUGE-L `0.2976671870474831`. A 38-entry evidence archive was transferred and matched SHA256 `239AA45FDE838E2820AAC3D8B3546B5A9AD827FCF5DD471499D4F4240A47E88A`. Audit found the summary still lacked acceptance-required post-warm-up throughput and runner-sampled process peak GPU memory. Commit `20c8b59` adds a pure-Python benchmark aggregator, a current-PID `nvidia-smi` sampler starting before step 68, completed-run integrity enforcement, and focused tests. Full Windows discovery now passes 60 tests with 15 environment-only skips; old run/archive remain preserved. | Commit this handoff update, push/sync `20c8b59`, run AutoDL tests, then rerun all 1,344 steps in a new benchmark namespace and create a versioned final evidence archive. |

| 2026-08-03 | `0f53a93` synchronized on Windows, GitHub, and AutoDL; benchmark smoke artifacts untracked | AutoDL benchmark instrumentation validation is complete. The first full AutoDL discovery hit a preserved intermittent Jittor process crash, the isolated affected test passed, and retry 1 passed all 60 tests with 8 environment-only skips. A fresh CUDA smoke then passed steps 0 -> 68 with exact frozen-language and Projector optimizer invariants. Its post-warm-up benchmark was accepted and measured optimizer step 68 at 117.05001987412734 effective samples/s and 1119.2908150463427 target tokens/s, with current-process peak GPU memory 1562 MiB from 4 samples. | Preserve the smoke namespace, launch a new full 0 -> 1344 benchmark run in `training_benchmark_0f53a93`, verify all acceptance fields, then create a versioned final evidence archive without overwriting the original archive. |

| 2026-08-03 | `8b51001` synchronized; Phase 4B result docs dirty on Windows | The new full benchmark run finished with exit code 0 and `passed=true`: steps 0 -> 1344, frozen/hash-unchanged GPT-2, exact Projector-only scope, finite updates, held-out NLL `6.643716599545368 -> 2.413187829110486`, 1,277 post-warm-up updates at 132.92145717737577 effective samples/s and 1413.1492154945145 target tokens/s, and 3058 MiB current-process peak GPU memory. The technical acceptance gap is closed. | Commit/push/sync the result docs, create `VisionZip-Jittor-phase4b-evidence-final-v2-20260803.tar.gz` without overwriting the original archive, verify its SHA256 on both hosts, then mark Phase 4B complete. |
| 2026-08-03 | `b943ab5` synchronized; Phase 4B completion docs dirty on Windows | Result documentation was committed and synchronized. Final v2 archive `VisionZip-Jittor-phase4b-evidence-final-v2-20260803.tar.gz` was created with 45 files, internal `SHA256SUMS` passed, copied to Windows, and matched AutoDL/Windows SHA256 `2EFEEAA88F18AB11B8431A7DD810B296366073D14B5717D02C72152DBA70C032`. The original archive remains preserved. Phase 4B is complete. | Commit/push/sync the completion status, then create a versioned next-phase scope and acceptance plan before implementation. |

| 2026-08-03 | `577dd56` baseline; Phase 5A source/config/docs dirty on Windows and AutoDL | Defined Phase 5A KV-cache scope; implemented per-layer native Jittor cache and cached greedy decode; added focused tests and the real GPT-2/Phase-4B-Projector/Phase-2-reference runner. Focused GPT-2 tests pass 8/8; current AutoDL full discovery passes 63 tests with 8 skips. Smoke 1 exposed a raw-vs-canonical Phase 4B config hash mismatch. Retry 1 passed, but audit found its nominal decode-only timing still included prefill. After moving prefill/cache construction outside the decode-only timer, corrected retry 2 passed with exact token IDs, per-step logits allclose at `1e-5`, 12 cache layers of `[1,12,78,64]`, prefill/decode-only `4.1622/7.3734 ms`, cached/uncached totals `15.9335/18.7474 ms`, speedup `1.17660x`, and 1122 MiB peak process GPU memory. | Rerun full tests after the timing correction, review/stage explicit files, commit/push/sync a clean source baseline, then run the full 64/128/192 formal benchmark in `tmux`; do not treat smoke timings as final evidence. |

| 2026-08-03 | `577dd56` baseline; Phase 5A implementation/docs remain dirty and uncommitted | Post-fix validation is complete for the development tree. Windows discovery passed 63 tests with 18 environment skips; AutoDL full discovery passed 63 tests with 8 skips on retry 2 after the decode-only timing correction. The preceding pure-LF retry hit the previously observed intermittent Jittor segfault in `test_frozen_stub_backpropagates_only_into_projector`; the isolated Phase 3 file then passed 2/2 and the full retry succeeded. Static `py_compile` and `git diff --check` pass. Phase 5A source hashes match between Windows and AutoDL for `gpt2.py`, its tests, config, and runner; the AutoDL plan document is older because the authoritative documentation edits remain on Windows. | Review and explicitly stage only the seven Phase 5A source/config/doc files, commit/push/sync a clean source baseline, rerun clean-commit tests, then launch the formal 64/128/192 benchmark in `tmux`. Preserve all smoke, segfault, isolated-test, and retry logs; do not treat dirty-tree timings as final evidence. |

| 2026-08-03 | `6d3ba71` synchronized; first clean formal benchmark and diagnostics preserved on AutoDL | Formal 64/128/192 run completed from a clean tracked tree. All 9/9 cached/uncached greedy token sequences and all cache/model/projector invariants were exact, and measured speedups were 1.05746x, 1.08881x, and 1.27338x. Top-level `passed=false` solely because strict raw-logit `atol=rtol=1e-5` failed for budget 64 sample 1 and all budget 192 samples; global max error was 0.0028228759765625. Layerwise diagnosis attributes the drift to CUDA FP32 shape-dependent GEMM/reduction rounding, and PyTorch 2.1.2 CUDA reproduced exact IDs with 7 strict failures and max error 0.001659393310546875. Representative Jittor softmax probability max error was 2.430344259174e-09 and allclose at 1e-5. | Preserve the failed namespace; implement a versioned dual-layer report where raw/centered logits remain diagnostics and exact IDs + exact cache + probability allclose form acceptance, add tests/docs, commit/sync, and rerun under a new namespace. |

| 2026-08-03 | `6d3ba71` baseline; acceptance-v2 source/config/docs dirty on Windows and AutoDL | The preserved 9-sample acceptance-v2 dirty smoke completed with `invariants_passed=true` but `passed=false`. All 9/9 token sequences and cache contracts remained exact. Coordinatewise stable-softmax `allclose(1e-5,1e-5)` failed in 6/9 samples, with probability max absolute error `2.7548452180448102e-05`; reconstructing total variation from the saved per-step reports gave a global maximum `3.428262002090551e-05`. A new PyTorch 2.1.2 CUDA diagnostic on budget 192/sample 1 also kept exact IDs, reproduced raw drift (`0.001659393310546875`), and measured max total variation `1.4558119111568148e-05`. This shows the v2 coordinatewise gate is still overly path-sensitive rather than a cache-semantic failure. | Preserve `acceptance_v2_dirty_smoke.*` and both PyTorch logs; implement acceptance-v3 with coordinatewise probability allclose retained as a diagnostic and a versioned per-step total-variation bound plus exact IDs/cache as the gate, add tests/docs, and rerun under a new namespace before committing. |

| 2026-08-03 | `6d3ba71` baseline; acceptance-v3 source/config/tests/docs dirty on Windows and AutoDL | Acceptance-v3 development validation is complete. The first `acceptance_v3_dirty_smoke.*` wrapper failed before model execution because its temporary JSON was not created; that namespace is preserved. Corrected `acceptance_v3_dirty_smoke_retry1.*` passed all 9/9 samples with exact greedy IDs, exact cache contracts, `invariants_passed=true`, top-level `passed=true`, and global maximum per-step total variation `3.688904192621437e-05` under the frozen `5e-5` bound. Coordinatewise diagnostics still failed as expected and remain diagnostic-only. AutoDL full retry 2 passed 74 tests with 8 skips after preserving the CRLF wrapper failure, intermittent Jittor segfault, and isolated 2/2 retry. | Finish Windows/static checks, explicitly stage the seven reviewed acceptance-v3 files, commit/push/sync, run clean-commit AutoDL tests, then launch the formal 3-warmup/10-run benchmark in a new commit-specific `tmux` namespace. |

| 2026-08-03 | `7227717` synchronized on Windows/GitHub/AutoDL; tracked AutoDL source clean | Acceptance-v3 clean validation passed. AutoDL full discovery: 74 tests, 8 skips, exit 0. Formal `kv_cache_benchmark_7227717.*` used 64/128/192 x samples 0/1/2, 32 tokens, 3 warmups, 10 measured runs; all 9/9 exact IDs/cache/TV gates passed, global max TV `3.505422367609121e-05`, GPT-2 and Projector were frozen/hash-unchanged, `invariants_passed=true`, top-level `passed=true`. Mean cached/uncached totals and speedups were 240.8540/243.3399 ms (1.01032x), 241.4139/255.4643 ms (1.05820x), and 241.0673/290.8731 ms (1.20661x); peak process GPU memory was 1168/1250/1322 MiB. | Commit/sync final-result docs, create an internally hashed Phase 5A evidence archive, copy it to Windows, verify matching outer SHA256, record completion, and only then begin the next phase. |

| 2026-08-03 | `a0d32de` completion documentation synchronized on Windows/GitHub/AutoDL; Phase 5A closed | `VisionZip-Jittor-phase5a-evidence-20260803.tar.gz` contains 112 internally checksummed entries, independently passed 112/112 unpacked checksum verification, is 521499 bytes, and has matching outer SHA256 `20093fb7550d6e17fc96566191236bc3631952998a5d4097d245ae1f2037ec81` on both hosts. It contains reviewed source/config/tests/docs, complete Phase 5A logs including preserved failures/diagnostics, formal `7227717` evidence, clean tests, and Git/environment metadata; large hashed model/data artifacts are deliberately excluded. The first AutoDL pull failed with the known transient GnuTLS termination error, and the `/etc/network_turbo` retry fast-forwarded successfully. Phase 5A is complete. | Design and review a bounded Phase 5B scope document before any Phase 5B implementation; preserve the Phase 5A archive and every historical log namespace unchanged. |

| 2026-08-03 | `2fa5d31` baseline; Submission Readiness changes dirty and uncommitted on Windows | Added a submission-oriented README rewrite, corrected environment contract, compact Phase 2/4B/5A result tables, full compact Phase 4B training and validation traces, reproducible figure/asset generation, Loss/LR/alignment/KV-cache/token-visualization PNGs, training/claim-boundary and readiness documents, dependency metadata, and focused tests. Source values are reconstructed from externally stored Phase 2/4A/4B/5A evidence archives with recorded SHA256 values. Phase 5B is deferred. | Review script arguments, links, images, and claims; run focused/full tests, py_compile, asset rebuild, link/image/diff checks; explicitly stage reviewed files; commit/push; sync and validate on AutoDL. |

| 2026-08-03 | `2fa5d31` baseline; Submission Readiness locally validated but still uncommitted on Windows | Review and local validation completed: focused 3/3 tests passed; Windows discovery ran 77 tests with 18 environment-only skips; compileall and `git diff --check` passed; regeneration from the four external evidence archives was byte-stable; 16 Markdown files passed relative-link checks; 5 PNGs passed integrity checks; reviewed text files are UTF-8 without BOM. No model/data/checkpoint artifacts are included. | Explicitly stage the reviewed allowlist, inspect the staged diff, commit/push, then fast-forward AutoDL and run clean-commit full tests and compileall without touching its untracked evidence artifacts. |

| 2026-08-03 | `7c1d45f` synchronized on Windows/GitHub/AutoDL; final validation note dirty on Windows | Submission-ready README, compact result tables, full training/validation traces, Loss/LR/alignment/KV-cache/token figures, environment/dependency guidance, claim boundary, readiness audit, asset builder, and focused tests were committed and pushed. AutoDL fast-forwarded cleanly. First full discovery hit the known intermittent Jittor segfault in the Phase 3 gradient-isolation test; isolated Phase 3 passed 2/2 and full retry 1 passed 77 tests with 8 skips. AutoDL compileall and diff checks passed. Attempt/retry logs are preserved untracked in `logs/submission_readiness/`; existing model/data/checkpoint artifacts remain untouched. | Commit/push this final validation note and fast-forward AutoDL; then perform a clean-environment README walkthrough, freeze the release/tag, recheck Jittor-Sprouts on submission day, and proceed to PPT/video. |

| 2026-08-03 | `3e484a2` synchronized baseline; clean README walkthrough preparation dirty on Windows | Audit found that `environment/activate_jittor.sh` hardcoded both the main checkout and the existing Jittor environment, so sourcing it from a clean clone would silently jump back to the main repository. The script now derives its project root from `BASH_SOURCE` and supports `VISIONZIP_JITTOR_ENV` and `VISIONZIP_CACHE_ROOT` overrides while preserving the documented defaults. README/environment guidance is updated accordingly. | Add a focused regression test, validate locally, commit/push/sync, then execute the clean-checkout/fresh-environment walkthrough and record evidence before tagging. |


| 2026-08-03 | `a02b9cf` synchronized; first clean walkthrough attempts preserved, README sample-input fix dirty on Windows | A fresh GitHub clone and fresh conda prefix installed all checked-in Jittor/Phase 3B/dev requirements and the editable package. The checkout-relative activation contract passed. The first orchestration attempt stopped on a walkthrough-only typo naming nonexistent `scripts/export_phase4b_clip_features.py`; the corrected help pass and fresh-env full discovery then passed 79 tests with 8 skips, compileall/diff checks passed, and synthetic PyTorch/Jittor alignment passed exactly. The real-CLIP step exposed a genuine README omission: generated sample PNGs are intentionally untracked, but the top-level README did not run `scripts/create_sample_images.py` before using `assets/sample_images`. README now includes that deterministic generation step and a regression test enforces ordering. | Validate and commit/push the README/test/handoff fix; create a new fixed-commit clone and fresh prefix; rerun the complete walkthrough, preserving both failed attempt logs as audit evidence. |

| 2026-08-03 | `04f098d` synchronized; clean walkthrough passed; walkthrough record locally validated and dirty on Windows | Final independent GitHub checkout and fresh conda-prefix walkthrough completed with `passed=true`: activation contract and 11-script help smoke passed; 80 tests passed with 8 environment-only skips; compileall/diff checks passed; synthetic alignment, real CLIP 64/128/192, native Jittor real GPT-2 smoke, and Phase 4B no-download preflight passed. The result is recorded in `docs/CLEAN_README_WALKTHROUGH.md` and `docs/results/clean_readme_walkthrough_04f098d.json`; external compact-summary SHA256 is `5b8ebf154a72d6c3951910330fcdadc250611fcaa1461c30fb5b70fa6cc03f01`. Windows review then passed focused 3/3 and full 80-test discovery with 18 environment-only skips, compileall/diff checks, 50 relative-link checks, and compact JSON integrity. No model/data/checkpoint/large-log artifact is being added. | Explicitly commit/push the reviewed allowlist, synchronize AutoDL, then create and publish immutable tag/release `v0.1.0-jittor-submission`; afterward proceed to PPT/video and submission-day list/claim review. |

| 2026-08-03 | `9671d36` synchronized on Windows/GitHub/AutoDL; final release-state handoff dirty on Windows | The clean walkthrough record, compact JSON, README/readiness links, and handoff update were explicitly staged, passed cached diff checks, committed as `9671d36 docs: record clean README walkthrough`, pushed to GitHub, and fast-forwarded on AutoDL. AutoDL retained all expected untracked Phase 3B/4B/5A/submission-readiness logs and outputs. The source tree is ready for immutable release freeze; no large artifact is included. | Commit/push this final handoff state, confirm the release tag is absent and all tracked trees are synchronized, then create/push annotated tag `v0.1.0-jittor-submission` and publish the GitHub Release without binary assets. |

When adding a new row, keep older rows. The newest row should state the exact commit or dirty-worktree state, the verified result, and the next blocking action.
