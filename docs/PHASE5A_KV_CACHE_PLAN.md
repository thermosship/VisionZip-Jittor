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

## 4. Correctness gates

For every configured budget and sample:

1. cached and uncached greedy token IDs must be exactly equal;
2. every compared next-token logit vector must satisfy `numpy.allclose` with `atol=1e-5` and `rtol=1e-5`;
3. cache layer count, shape, and final cached length must match the GPT-2 configuration and generated sequence length;
4. GPT-2 parameters must remain stop-grad and SHA256-identical before and after the run;
5. Projector parameters must remain stop-grad and SHA256-identical before and after the run;
6. GPT-2 artifact, checkpoint, feature-manifest, and Phase 2 reference identity checks must pass.

A speedup is measured and reported but is not a pass/fail requirement. Small workloads can be dominated by framework synchronization, kernel launch, and token transfer overhead.

## 5. Benchmark protocol

The checked-in configuration is `configs/phase5a_kv_cache.json`:

- budgets: 64, 128, 192;
- sample rows: 0, 1, 2 from each matching Phase 2 reference;
- generated tokens per sample: 32, with EOS stopping disabled so paths perform the same amount of work;
- warm-up runs: 3;
- measured runs: 10;
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
  --output-json logs/phase5a/kv_cache_benchmark.json
```

The final JSON contains configuration and artifact hashes, per-budget/per-sample correctness traces, raw timing samples, aggregate latency/throughput statistics, process GPU-memory samples, frozen-model invariants, and a top-level `passed` field.

Long AutoDL runs must be launched in `tmux`. Generated logs and outputs remain outside normal Git history and belong in the Phase 5A evidence archive.

## 7. Claim boundary

Phase 5A may establish that native Jittor GPT-2 cached decoding is numerically aligned with this repository's uncached oracle and may report measured latency, throughput, and memory for the pinned RTX 4090 environment.

It must not be described as:

- improved caption quality;
- a larger-language-model result;
- a mixed-precision result;
- a comparison against human-caption or multi-reference COCO metrics;
- proof of a universal speedup on every device, token length, or Jittor version.

The Phase 4B quality boundary remains unchanged: its single BLIP-2 synthetic reference metrics are not directly comparable with multi-reference COCO numbers.

## 8. Completion checklist

- [x] Implement per-layer native Jittor GPT-2 KV cache.
- [x] Preserve and test the uncached execution path.
- [x] Add focused cached/full-recompute parity unit tests.
- [x] Run the initial real GPT-2 CUDA cache diagnostic.
- [ ] Add and commit the formal Phase 5A runner and configuration.
- [ ] Run all tests from a clean synchronized commit on AutoDL.
- [ ] Run the formal correctness and benchmark protocol in `tmux`.
- [ ] Record final JSON/logs, source commit, environment, and hashes.
- [ ] Build the Phase 5A evidence archive and verify matching AutoDL/Windows SHA256 values.
- [ ] Update this document and README with the measured final results.


## 9. Implementation status (2026-08-03, in progress)

Implemented in the current development worktree:

- per-layer native Jittor GPT-2 key/value caching;
- cached attention with past-aware causal masking and position IDs;
- cached greedy generation while preserving the uncached oracle;
- cache layer/shape/length validation;
- focused cached-vs-uncached unit tests;
- a versioned runner that loads the real GPT-2 artifact, Phase 4B final Projector checkpoint, and Phase 2 64/128/192 real-CLIP references;
- provenance/hash/frozen-parameter checks, correctness traces, repeated timing summaries, throughput, speedup, and process GPU-memory sampling.

Validated development results:

```text
Focused native GPT-2 tests: 8 passed
Full AutoDL discovery after decode-only timing fix: 63 passed, 8 skipped (retry 2)
Reduced runner smoke: passed=true
Smoke scope: budget 64, sample index 0, max_new_tokens=2,
             warmup_runs=0, measured_runs=1
Cached/uncached token IDs: exact
Per-step logits: allclose at atol=rtol=1e-5
Final cache shape: 12 layers of [1, 12, 78, 64]
Cached prefill mean: 4.1622 ms
Cached decode-only mean: 7.3734 ms
Cached total mean: 15.9335 ms
Uncached total mean: 18.7474 ms
Uncached/cached total speedup: 1.17660x
Peak process GPU memory: 1122 MiB
```

The first runner smoke correctly failed on `phase4b_config_hash_exact: false` because it compared raw file bytes with the checkpoint's canonical JSON hash. The runner now uses `canonical_json_sha256(phase4b_config.to_dict())`; retry 1 passed. Review then found that the nominal decode-only benchmark still performed prefill inside the timed function. The runner now precomputes an immutable initial cache and first token outside the decode-only timer; corrected retry 2 passed. All earlier logs are retained. After the timing correction, a new full AutoDL discovery passed 63 tests with 8 environment-only skips on retry 2. The preceding pure-LF retry hit the previously recorded intermittent Jittor process segfault in `test_frozen_stub_backpropagates_only_into_projector`; the isolated Phase 3 test file then passed 2/2, and the complete retry passed. The failed and successful logs remain separate.

This is not the final Phase 5A result. The successful smoke ran from an uncommitted development tree whose HEAD was `577dd56`. The full benchmark must run after an explicit source commit and clean synchronization, using all three budgets, all three samples, 32 new tokens, 3 warmups, and 10 measured runs. Final documentation and evidence packaging remain pending.
