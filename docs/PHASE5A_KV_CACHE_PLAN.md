# Phase 5A: Native Jittor GPT-2 KV-cache generation

## 1. Objective

Phase 5A adds native Jittor GPT-2 key/value caching for autoregressive greedy decoding and records a reproducible cached-versus-uncached decode benchmark. It closes the explicit Phase 4 limitation that generation recomputed the full multimodal prefix at every output token.

This phase is an inference/runtime milestone. It does not train CLIP, VisionZip, GPT-2, or the Phase 4B Projector, and it does not claim improved caption quality.

## 2. Fixed source artifacts

The formal run uses:

- the real GPT-2 small artifact exported in Phase 3B under `outputs/phase3b/gpt2`;
- the completed Phase 4B Projector checkpoint `outputs/phase4b/commoncatalog_cc_by_8k/training_benchmark_0f53a93/best_projector.npz`;
- the Phase 4B feature manifest as checkpoint-provenance evidence;
- the three deterministic Phase 2 real-CLIP references at nominal budgets 64, 128, and 192.

The Projector was trained at nominal budget 64, but its token-wise `1024 -> 768 -> 768` mapping accepts all three Phase 2 token counts. Cross-budget generation is therefore a runtime-path validation, not a claim that the Phase 4B training result generalized optimally to larger budgets.

## 3. KV-cache contract

Each GPT-2 layer returns a key/value pair with shape:

```text
[batch, attention_heads, cached_tokens, head_dim]
```

For cached calls:

- new position IDs begin at `cached_tokens`;
- the attention key length is `cached_tokens + query_tokens`;
- the causal mask has shape `query_tokens x total_key_tokens`;
- an optional attention mask must cover the complete cached-plus-query sequence;
- all layers must have the same cached length;
- cached generation performs one full-prefix prefill and then one-token decode calls.

The original uncached `execute()` path remains unchanged and is the correctness oracle.

## 4. Versioned numerical contract

### 4.1 Acceptance v1 and the preserved formal failure

Commit `6d3ba713476b054f9cf9ea374144de538b2ec601` used a single strict raw-logit gate: every cached/full-recompute next-token logit vector had to satisfy `numpy.allclose(atol=1e-5, rtol=1e-5)`. The clean formal run preserved exact greedy IDs for all 9/9 samples and passed every cache, provenance, frozen-parameter, and hash invariant, but top-level `passed=false` because long-sequence raw logits exceeded that gate.

The failed result is retained under `logs/phase5a/kv_cache_benchmark_6d3ba71.*`. It must not be overwritten or retroactively relabeled as successful.

### 4.2 Why raw-logit strict equality is diagnostic-only

Layerwise diagnosis found that cached and uncached paths begin with identical current-token embeddings. Their CUDA FP32 arithmetic then uses different matrix and reduction shapes: cached decode has query length 1, while the oracle recomputes the complete prefix. Small shape-dependent rounding differences accumulate through 12 transformer layers and the historical cache.

This behavior is not unique to Jittor. The official Hugging Face PyTorch GPT-2 path on the same RTX 4090, using PyTorch `2.1.2+cu118`, also produced exact cached/full-recompute greedy IDs while failing strict raw-logit `1e-5` on 7 steps, with maximum absolute error `0.001659393310546875`.

For the representative Jittor budget-192/sample-1/step-20 diagnostic:

```text
raw logits max absolute error:       5.531311035156e-04
centered logits max absolute error:  9.297727791235e-05
softmax probability max error:       2.430344259174e-09
softmax probability allclose 1e-5:   true
greedy token ID:                     exact
```

Much of the raw-logit difference is close to a vocabulary-wide common shift, which does not change softmax probabilities. Centered logits are useful for diagnosis but are not an acceptance gate.

### 4.3 Acceptance v2 and the preserved dirty-smoke failure

Acceptance v2 retained raw and centered logits as diagnostics and attempted to gate stable float64 softmax probabilities with coordinatewise `numpy.allclose(atol=1e-5, rtol=1e-5)`. The complete 9-sample reduced-timing dirty smoke is preserved under `logs/phase5a/acceptance_v2_dirty_smoke.*`. It kept all 9/9 greedy token sequences and all cache/model/projector invariants exact, but top-level `passed=false`: 6/9 samples had one or more coordinatewise probability failures.

The largest coordinate probability error was `2.7548452180448102e-05`. The saved per-step mean absolute errors imply a maximum total variation distance of `3.428262002090551e-05`, where

```text
TV(p, q) = 0.5 * sum_i |p_i - q_i|
```

A new PyTorch diagnostic on budget 192/sample 1 likewise kept exact token IDs, reproduced the raw-logit maximum error `0.001659393310546875`, and measured maximum total variation `1.4558119111568148e-05`. Coordinatewise allclose is therefore retained as useful evidence, but it is too sensitive to individual vocabulary coordinates to define cache-semantic equivalence for this pinned CUDA FP32 path. The v2 failure remains a failed result and is not overwritten or relabeled.

### 4.4 Acceptance v3

Acceptance v3 freezes a distribution-level bound before the next clean formal retry. For every configured budget and sample it requires:

1. cached and uncached greedy token IDs are exactly equal;
2. the stable-float64 softmax distributions have per-step total variation distance no greater than `5e-5`;
3. cache layer count, shape, and final cached length exactly match the GPT-2 configuration and generated sequence length;
4. GPT-2 parameters remain stop-grad and SHA256-identical before and after the run;
5. Projector parameters remain stop-grad and SHA256-identical before and after the run;
6. GPT-2 artifact, checkpoint, feature-manifest, and Phase 2 reference identity checks pass.

The `5e-5` bound means that no generation step may move more than `0.005%` of probability mass between the two distributions. It is above the complete v2 dirty-smoke maximum (`3.428262002090551e-05`) and the matching PyTorch diagnostic (`1.4558119111568148e-05`) while remaining an absolute, vocabulary-size-independent distribution bound. It is specific to this pinned Phase 5A protocol and is not claimed as a universal framework tolerance. The threshold is frozen before the clean-commit retry and must not be changed in response to that retry.

The runner continues to record raw logits, centered logits, and coordinatewise softmax-probability maximum/mean errors, strict `1e-5` allclose status, and failed-step indices as `diagnostic_only`. It additionally records per-step total variation, top-1 probability error, top-1 margins, and argmax agreement. Only exact IDs, the total-variation bound, exact cache structure, and the model/provenance invariants control `passed`.

A speedup is measured and reported but is not a pass/fail requirement. Small workloads can be dominated by framework synchronization, kernel launch, and token transfer overhead.

## 5. Benchmark protocol

The checked-in configuration is `configs/phase5a_kv_cache.json`:

- schema: `phase5a_kv_cache_benchmark_config_v3`;
- budgets: 64, 128, 192;
- sample rows: 0, 1, 2 from each matching Phase 2 reference;
- generated tokens per sample: 32, with EOS stopping disabled so paths perform the same amount of work;
- warm-up runs: 3;
- measured runs: 10;
- raw-logit diagnostic tolerance: `atol=rtol=1e-5`;
- centered-logit diagnostic tolerance: `atol=rtol=1e-5`;
- coordinatewise softmax-probability diagnostic tolerance: `atol=rtol=1e-5`;
- per-step total-variation acceptance bound: `5e-5`;
- current-process GPU memory sampling interval: 0.1 seconds.

The formal runner records four scopes separately:

- cached prefill latency;
- cached decode-only latency after prefill;
- cached total generation latency;
- uncached full-recompute total generation latency.

Reported decode throughput uses the number of one-token cached decode calls (`max_new_tokens - 1`). Total-generation throughput uses all generated tokens. Each timed region calls `jt.sync_all()` at its boundaries.

## 6. Runner

```bash
python scripts/run_phase5a_kv_cache.py \
  --device cuda \
  --output-json logs/phase5a/kv_cache_benchmark_<short-commit>.json
```

The v3 JSON contains configuration and artifact hashes, per-budget/per-sample raw/centered/coordinatewise-probability diagnostics, total-variation acceptance reports, explicit acceptance fields, raw timing samples, aggregate latency/throughput statistics, process GPU-memory samples, frozen-model invariants, and a top-level `passed` field.

Long AutoDL runs must be launched in `tmux`. Generated logs and outputs remain outside normal Git history and belong in the Phase 5A evidence archive. Every formal retry must use a new commit-specific namespace.

## 7. First clean formal benchmark: preserved failed result

Source commit and protocol:

```text
source commit: 6d3ba713476b054f9cf9ea374144de538b2ec601
tracked worktree clean: true
budgets: 64, 128, 192
samples per budget: 3
new tokens: 32
warmups / measured runs: 3 / 10
top-level passed: false
invariants_passed: true
exact greedy token sequences: 9 / 9
```

Performance measurements from that run remain valid runtime observations even though acceptance v1 failed:

| Budget | Cached prefill mean | Cached decode-only mean | Cached total mean | Uncached total mean | Speedup | Peak process GPU memory |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 3.1244078030 ms | 223.5921872159 ms | 229.8019619038 ms | 243.0066477507 ms | 1.0574611537x | 1168 MiB |
| 128 | 3.4648358822 ms | 222.9951935510 ms | 233.2324676216 ms | 253.9463482797 ms | 1.0888121661x | 1250 MiB |
| 192 | 3.8958571851 ms | 222.1120735010 ms | 230.9646122158 ms | 294.1047631204 ms | 1.2733758661x | 1322 MiB |

Raw-logit v1 outcome:

- budget 64: sample 1 failed at step 31; budget maximum error `0.0006313323974609375`;
- budget 128: all 3 samples passed the original raw-logit gate;
- budget 192: all 3 samples had strict failed steps; budget/global maximum error `0.0028228759765625`.

These numbers do not become a successful Phase 5A result merely because later acceptance contracts are better specified. The acceptance-v2 dirty smoke also remains a failed result. A new acceptance-v3 clean-commit full run is required.

## 8. Acceptance-v3 development validation

The complete 9-sample reduced-timing dirty smoke finished on 2026-08-03 from the `6d3ba71` development worktree with the acceptance-v3 source/config changes present but not yet committed. The first wrapper attempt is preserved as `acceptance_v3_dirty_smoke.*`; it exited before model execution because a malformed temporary-script newline prevented `/tmp/phase5a_acceptance_v3_dirty_smoke.json` from being created. The corrected retry is preserved separately as `acceptance_v3_dirty_smoke_retry1.*` and recorded:

```text
budgets / samples:                 64, 128, 192 / 0, 1, 2
new tokens:                       32
warmups / measured runs:          0 / 1
exact greedy token sequences:     9 / 9
exact cache contracts:            9 / 9
TV within frozen 5e-5 bound:      9 / 9
global maximum TV:                3.688904192621437e-05
invariants_passed:                true
top-level passed:                 true
```

The coordinatewise diagnostics intentionally remain capable of failing without rejecting a trace: raw-logit allclose failed in 4/9 samples, centered-logit allclose failed in 9/9, and coordinatewise softmax-probability allclose failed in 6/9. Exact greedy IDs, exact cache contracts, and the distribution-level TV gate nevertheless passed every sample. The largest dirty-smoke raw-logit error was `0.004314422607421875`; because raw logits are diagnostic-only, this does not become a claim of strict raw-logit equality.

Development validation also includes 11 focused Phase 5A metric tests and full Windows discovery (`74` tests, `18` skipped). On AutoDL, the final full retry passed `74` tests with `8` environment skips. Earlier AutoDL wrappers are retained: one had a CRLF-induced shell exit-reporting error after all tests passed, and another hit the previously observed intermittent Jittor segfault in `test_frozen_stub_backpropagates_only_into_projector`; the isolated Phase 3 retry passed 2/2 before the final full retry succeeded.

This result validates the acceptance-v3 implementation but is not formal evidence because the tracked worktree was dirty and timing used zero warmups plus one measured run. A new clean commit, clean-commit tests, and the full 3-warmup/10-measured-run benchmark remain required.

## 9. Claim boundary

Phase 5A may establish exact greedy decisions, exact cache-contract behavior, probability-level cached/full-recompute alignment, and reproducible latency/throughput/memory measurements for the pinned RTX 4090 environment.

It must disclose the strict raw-logit diagnostics and the matching PyTorch CUDA baseline. It must not be described as:

- raw-logit bitwise equality or universal strict-`1e-5` raw-logit equality;
- improved caption quality;
- a larger-language-model result;
- a mixed-precision result;
- a comparison against human-caption or multi-reference COCO metrics;
- proof of a universal speedup on every device, token length, or Jittor version.

The Phase 4B quality boundary remains unchanged: its single BLIP-2 synthetic reference metrics are not directly comparable with multi-reference COCO numbers.

## 10. Completion checklist

- [x] Implement per-layer native Jittor GPT-2 KV cache.
- [x] Preserve and test the uncached execution path.
- [x] Add focused cached/full-recompute parity unit tests.
- [x] Add and commit the v1 formal Phase 5A runner and configuration.
- [x] Run all v1 tests from clean synchronized commit `6d3ba71` on AutoDL.
- [x] Run and preserve the first clean formal benchmark and its failed v1 result.
- [x] Diagnose the raw-logit drift layer-by-layer and reproduce the same class of drift with PyTorch CUDA.
- [x] Implement and preserve acceptance-v2 raw/centered/probability reports and its failed 9-sample dirty smoke.
- [x] Diagnose all v2 failures, compute the complete Jittor total-variation envelope, and add a matching PyTorch distribution diagnostic.
- [x] Implement acceptance-v3 total-variation reports and focused pure-NumPy tests in the current development tree.
- [x] Pass the acceptance-v3 9-sample dirty smoke under a new namespace.
- [ ] Commit and synchronize acceptance v3.
- [ ] Run all tests from the new clean synchronized commit on AutoDL.
- [ ] Run the acceptance-v3 formal protocol in a new commit-specific `tmux` namespace.
- [ ] Audit the final JSON/logs, source commit, environment, and hashes.
- [ ] Build the Phase 5A evidence archive and verify matching AutoDL/Windows SHA256 values.
- [ ] Update this document and README with the passing retry and final archive hash.

## 11. Immediate next action

Run final Windows tests/static checks, explicitly stage only the reviewed acceptance-v3 source/config/test/document files, commit and synchronize the new source baseline, and then run clean-commit AutoDL tests. The formal retry must use a new commit-specific `logs/phase5a/kv_cache_benchmark_<short-commit>.*` namespace with 3 warmups and 10 measured runs, and it must preserve every v1/v2/dirty-smoke failure and diagnosis file.
