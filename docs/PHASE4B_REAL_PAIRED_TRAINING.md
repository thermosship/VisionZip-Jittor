# Phase 4B: Licensed Real Paired-Data Projector Training

## 1. Status and scope

The benchmark-instrumented Phase 4B CUDA acceptance run completed successfully on
**2026-08-03** using an NVIDIA GeForce RTX 4090 and Jittor 1.3.11.0. The run
summary records both `passed: true` and `completed_training: true`.

Phase 4B advances the Phase 4A three-pair infrastructure smoke to a licensed,
fixed, externally sourced pilot:

```text
8,192 pinned CommonCatalog CC-BY image/caption pairs
                    |
                    v
frozen CLIP ViT-L/14-336 + exact VisionZip budget 64 features
                    |
                    v
trainable native Jittor Projector (1024 -> 768 -> 768)
                    |
                    v
frozen native Jittor GPT-2 small (124,439,808 parameters)
                    |
                    v
target-only causal loss, deterministic held-out evaluation, checkpoints/resume
```

Only the 1,377,792-parameter Projector is optimized. CLIP/VisionZip features are
precomputed and GPT-2 remains frozen and hash-identical before and after
training.

At the time this result document was first committed, the final versioned v2
evidence archive was still being assembled. Phase 4B should be labeled complete
only after that archive has been copied to Windows and its SHA256 matches on both
hosts.

## 2. Reproducible inputs

### 2.1 Dataset

| Field | Value |
|---|---|
| Dataset | `common-canvas/commoncatalog-cc-by` |
| Pinned revision | `80f50fe4a1ca937f37a11be3f8eee5199d776ff3` |
| Materialized samples | 8,192 |
| Train / held-out split | 7,168 / 1,024 |
| Split seed | 2026 |
| Caption/reference | one `blip2_caption` synthetic reference per image |
| Prepared `samples.jsonl` SHA256 | `c2b205622a67349ee22547e2648c264cd6a71979906d57cc4b59fede10e330f2` |

The prepared manifest retains source coordinates, creator/page attribution,
row-level Creative Commons license URLs, upstream source identity hashes, and
independently computed embedded-JPEG hashes. The five pinned source Parquet
objects total 1,263,965,106 bytes.

### 2.2 Frozen visual features

| Field | Value |
|---|---|
| Vision model | `openai/clip-vit-large-patch14-336` |
| Pinned model revision | `ce19dc912ca5cd21c8a653c79e251e808ccabcd1` |
| VisionZip preset | 54 dominant + 10 contextual + CLS |
| Stored token shape | `[N, 65, 1024]`, float32 |
| Feature shards | 32 x 256 samples |
| Feature manifest SHA256 | `2673aafd3ec7084c7eae54cd8eaac693fc21f84892cccf60a0c14f8c349a36a9` |
| Phase 4B config SHA256 | `ff80fda5bfc9a56580ccd26ede7cf4f3a8ea3742c10c4d6f55dc5afa8dbbe6ac` |

All 8,192 feature rows were independently checked for exact sample order,
finite float32 values, valid discrete index arrays, and shard SHA256 integrity.

## 3. Training schedule

| Setting | Value |
|---|---:|
| Micro-batch | 4 |
| Gradient accumulation | 4 |
| Effective samples/update | 16 |
| Optimizer | Jittor Adam, Projector parameters only |
| Optimizer steps | 1,344 |
| Learning rate | 1e-4 |
| Warm-up | 67 optimizer steps |
| Post-warm-up schedule | cosine decay |
| Evaluation/checkpoint interval | 112 steps |
| Rolling checkpoints | last 4, plus best and final |
| Seed | 2026 |

The loss for an accumulated optimizer update is weighted by target-token count,
not by an unweighted mean of micro-batch losses. Checkpoints restore Projector,
Adam moments and step, scheduler position, deterministic epoch/order cursor, and
RNG state.

## 4. Validated execution

### 4.1 Source and environment

| Field | Value |
|---|---|
| Training source commit | `8b510017e4250bc626ca001bcf56cc05b7e09fa9` |
| Benchmark implementation commit | `20c8b59` |
| Run directory | `outputs/phase4b/commoncatalog_cc_by_8k/training_benchmark_0f53a93` |
| Log directory | `logs/phase4b/training_benchmark_0f53a93` |
| Runtime | Jittor 1.3.11.0, CUDA |
| GPU | NVIDIA GeForce RTX 4090, 24,564 MiB |
| Driver | 580.105.08 |
| Wall-clock interval | 2026-08-03 14:14:33 to 14:19:51 +08:00 |

### 4.2 Integrity and optimization checks

| Check | Result |
|---|---|
| Top-level passed | `true` |
| Completed all configured training | `true` |
| Optimizer steps | `0 -> 1344` |
| Adam `n_step` | `1344` |
| All updates finite | `true` |
| GPT-2 all stop-grad | `true` |
| GPT-2 unchanged | `true` |
| Optimizer scope exactly Projector-only | `true` |
| Projector trainable after evaluation | `true` |
| Projector maximum parameter delta | `0.012454323470592499` |
| Benchmark evidence accepted | `true` |

A corrected fresh smoke passed steps 0 -> 2, and an explicit checkpoint resume
passed steps 2 -> 4 with the same optimizer-step ordering and all frozen-model
invariants. The later benchmark smoke also passed steps 0 -> 68 and measured the
first post-warm-up update before the full run was launched.

## 5. Held-out results

Primary metrics use all 1,024 held-out samples and 10,744 target tokens.

| Metric | Initial | Final | Change |
|---|---:|---:|---:|
| Target NLL | 6.643716599545368 | 2.413187829110486 | 63.6771407546851% reduction |
| Target perplexity | 767.9438354472138 | 11.169510943754625 | 68.7535774228867x lower |

The best validation checkpoint and final checkpoint are both at optimizer step
1,344.

The deterministic 128-sample generation subset recorded:

| Metric | Value |
|---|---:|
| BLEU-1 | 0.28304947283049475 |
| BLEU-4, add-one smoothing | 0.05727683512769526 |
| ROUGE-L mean F1 | 0.26176086973563917 |

These generation metrics use lowercase Unicode word tokenization and exactly one
BLIP-2 synthetic reference per image.

## 6. Post-warm-up benchmark evidence

The benchmark excludes model loading, startup, evaluation, checkpoint I/O, and
generation from optimizer-step throughput. GPU memory is sampled separately for
the current Python PID across the post-warm-up training loop, including periodic
evaluation/checkpoint work.

| Field | Result |
|---|---:|
| Warm-up optimizer steps | 67 |
| Measured step range | 68 -> 1344 |
| Measured optimizer steps | 1,277 |
| Mean optimizer-step compute | 120.37183717184918 ms |
| Effective sample count | 20,432 |
| Effective samples/s | 132.92145717737577 |
| Target token count | 217,222 |
| Target tokens/s | 1,413.1492154945145 |
| Current-process peak GPU memory | 3,058 MiB |
| GPU memory samples | 1,416 at 0.1 s interval |

The throughput is a training-compute measurement for this fixed RTX 4090/Jittor
configuration, not an end-to-end data-preparation or production-serving claim.

## 7. Tests and preserved failures

- Windows full discovery: 60 tests passed, 15 environment-only skips.
- The first AutoDL full-discovery attempt hit an intermittent Jittor process
  crash in `test_frozen_stub_backpropagates_only_into_projector`; its log is
  preserved rather than deleted.
- The isolated affected test passed.
- AutoDL full-discovery retry 1 passed all 60 tests with 8 environment-only
  skips.
- The original Phase 4B eval-state smoke false negative is also preserved. Its
  optimizer updates were valid; the later source correction restored
  `projector.train()` after evaluation before checking trainability.

## 8. Artifact integrity

| Artifact | SHA256 |
|---|---|
| Benchmark training summary | `b13532ea76352ebe9fcb3e1c3e0a4c6018f7865f930adc21435503e6daff4b80` |
| Benchmark metrics JSONL | `f4ab48d472fb1cc74376e0661f3177c8f31d1f3e51eeb895f1173bdb32f6f0ca` |
| Final Projector checkpoint | `d209bba83b7ee822efa5e29912ce0bc82748fdcb90303f06981f912ee9f928fa` |

Generated dataset images, feature shards, logs, and checkpoints remain outside
normal Git history. They belong in the versioned evidence archive or the remote
artifact store.

## 9. Claim boundary

This phase proves that the native Jittor path can train a Projector on a real,
licensed, externally sourced paired pilot while preserving exact frozen GPT-2
and optimizer-scope invariants, and that reproducible held-out and benchmark
measurements can be recorded.

It does **not** prove:

- human-level or state-of-the-art image captioning;
- direct parity with multi-reference COCO metrics;
- end-to-end CLIP fine-tuning;
- mixed-precision correctness;
- KV-cache generation performance;
- quality scaling to a larger frozen language model.

The reported BLEU/ROUGE values are single-synthetic-reference measurements and
must not be directly compared with multi-reference COCO benchmark numbers.
