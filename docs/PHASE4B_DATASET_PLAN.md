# Phase 4B Licensed Paired-Dataset Plan

## Status

Phase 4B has started at the **versioned planning and data-infrastructure** boundary. This document fixes the source snapshot, subset, attribution policy, storage format, training schedule, evaluation protocol, and acceptance criteria **before** any large download or training run.

No external-dataset training result is claimed by this document. The first large operation must be the preflighted preparation command described below.

## 1. Objective and boundary

Phase 4A proved that precomputed real CLIP/VisionZip tokens can train only the native Jittor Projector while native Jittor GPT-2 remains frozen. Phase 4B replaces the three project-generated smoke pairs with a licensed external paired dataset and adds:

1. pinned source objects and row-level attribution;
2. deterministic train/held-out selection;
3. resumable dataset materialization;
4. hashed, sharded real-CLIP/VisionZip features;
5. multi-step Projector training with gradient accumulation and checkpoint retention;
6. held-out teacher-forced and generated-caption evaluation.

Phase 4B does **not** train CLIP or GPT-2. The intended optimizer scope remains Projector-only.

## 2. Dataset decision

### 2.1 Selected source

- Dataset: `common-canvas/commoncatalog-cc-by`
- Hugging Face revision: `80f50fe4a1ca937f37a11be3f8eee5199d776ff3`
- Published fields used by this project:
  - image: `jpg`;
  - synthetic caption: `blip2_caption`;
  - source status: `status`;
  - attribution/license: `licensename`, `licenseurl`, `unickname`, `pageurl`;
  - integrity/identity: `sha256`, `photoid`, `uid`.

Official sources:

- Dataset card: https://huggingface.co/datasets/common-canvas/commoncatalog-cc-by
- Pinned dataset API object: https://huggingface.co/api/datasets/common-canvas/commoncatalog-cc-by/revision/80f50fe4a1ca937f37a11be3f8eee5199d776ff3
- CommonCanvas paper: https://arxiv.org/abs/2310.16825
- CC BY 4.0 summary: https://creativecommons.org/licenses/by/4.0/

At the pinned revision, the Hugging Face dataset-size API reports 14,581,672 rows and 21,465,186,529,987 bytes of Parquet data. The complete dataset is therefore far larger than the 200 GiB AutoDL data disk and must not be downloaded wholesale.

### 2.2 Why this source

The dataset card identifies a CC-BY subset of CommonCatalog and provides row-level source/license metadata. This is a better provenance fit for the current reproduction than datasets whose image copyright status is not uniform or not carried into each sample manifest.

The captions are BLIP-2 synthetic captions rather than multiple human references. They are suitable for testing real paired-data optimization and deterministic held-out loss, but they do not justify a claim of human-caption quality.

### 2.3 License and attribution policy

The dataset card carries a `cc-by-4.0` tag, while individual rows may point to an original CC BY version such as CC BY 2.0. Phase 4B treats the **row-level `licenseurl` as authoritative** and accepts only URLs under:

- `http://creativecommons.org/licenses/by/`
- `https://creativecommons.org/licenses/by/`

Every prepared sample must preserve:

- creator/user name where supplied;
- title where supplied;
- Flickr source-page URL;
- original license name and URL;
- source image SHA256;
- dataset revision, source Parquet path, and row number.

Prepared images, per-sample JSONL, downloaded Parquet, feature shards, logs, and checkpoints are generated artifacts and must not enter normal Git history.

## 3. Fixed pilot subset

The versioned config is:

`configs/phase4b_commoncatalog_cc_by_8k.json`

It selects five exact Parquet objects from partitions 0 through 4 with:

- least-dimension bucket: 512–768;
- aspect-ratio bucket: 1–1;
- total source files: 5;
- total declared rows: 9,621;
- total declared source size: 1,263,965,106 bytes;
- target accepted pairs: 8,192;
- training pairs: 7,168;
- held-out pairs: 1,024;
- split seed: 2026.

The source objects and their Git repository OIDs, byte sizes, and Parquet row counts are committed in the config. The preparation script verifies local byte size and row count, then records the downloaded file SHA256 in the generated manifest.

### 3.1 Filtering

Rows are accepted only when all of the following hold:

1. `status == "success"`;
2. normalized BLIP-2 caption contains 3–40 whitespace-delimited words;
3. caption length is at most 256 characters;
4. metadata and decoded image dimensions are at least 336 × 336 on the shorter side;
5. row-level license URL is in the CC BY family allowed above;
6. embedded image decodes as JPEG;
7. decoded dimensions match metadata;
8. row `sha256` is a valid lowercase digest and has not already been accepted;
9. the materialized embedded JPEG receives its own independently computed SHA256.

CommonCatalog's row `sha256` and the SHA256 of the embedded Parquet JPEG are intentionally stored separately. The row digest is upstream source identity/provenance; the embedded image is a processed JPEG and therefore need not be byte-identical. Integrity verification uses the independently computed `image_sha256` for the materialized JPEG while retaining `source_image_sha256` for source deduplication and attribution provenance.

The accepted sequence follows pinned shard order and source-row order. The held-out set is then selected by ranking `SHA256("2026:" + sample_id)` and taking exactly 1,024 samples. This gives an exact, deterministic held-out count while preserving the original accepted order in the manifest.

The generated prepared manifest is:

- `datasets/phase4b/commoncatalog_cc_by_8k/manifest.json`
- `datasets/phase4b/commoncatalog_cc_by_8k/samples.jsonl`

Both are generated and ignored by Git.

## 4. Disk and execution preflight

The preparation command defaults to **preflight only**. It performs no dataset download unless `--execute` is present.

Estimated disk reservation:

| Component | Conservative estimate |
|---|---:|
| Pinned source Parquet | 1,263,965,106 bytes |
| Extracted selected JPEGs | up to 1,263,965,106 bytes |
| 8,192 × 65 × 1,024 float32 visual tokens | 2,181,038,080 bytes |
| Temporary/checkpoint/log headroom | 4 GiB |
| Total required by preflight | 9,003,935,588 bytes (~8.39 GiB) |

The estimate deliberately counts both cached Parquet and extracted images and does not rely on compression savings. The verified AutoDL disk had about 196 GiB free at Phase 4B start.

### AutoDL/Jupyter Terminal — preflight only

```bash
cd /root/autodl-tmp/VisionZip-Jittor
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/envs/visionzip-jittor
export USE_TORCH=0 USE_TF=0 USE_FLAX=0 OMP_NUM_THREADS=8
python scripts/prepare_phase4b_dataset.py \
  --config configs/phase4b_commoncatalog_cc_by_8k.json \
  --output-dir datasets/phase4b/commoncatalog_cc_by_8k
```

### AutoDL/Jupyter Terminal — materialize in tmux

Run only after the preflight JSON reports `disk_preflight_passed: true`.

```bash
tmux new -s phase4b-data
cd /root/autodl-tmp/VisionZip-Jittor
source /etc/network_turbo
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/envs/visionzip-jittor
export HF_HUB_DISABLE_XET=1
export HF_HUB_ETAG_TIMEOUT=60
export HF_HUB_DOWNLOAD_TIMEOUT=600
export HF_HOME=/root/autodl-tmp/cache/huggingface
export USE_TORCH=0 USE_TF=0 USE_FLAX=0 OMP_NUM_THREADS=8
set -o pipefail
python scripts/prepare_phase4b_dataset.py \
  --config configs/phase4b_commoncatalog_cc_by_8k.json \
  --output-dir datasets/phase4b/commoncatalog_cc_by_8k \
  --cache-dir /root/autodl-tmp/cache/huggingface \
  --execute \
  2>&1 | tee logs/phase4b/prepare_dataset.log
```

## 5. Visual feature precompute

The feature producer is:

`scripts/precompute_phase4b_features.py`

Pinned visual path:

- model: `openai/clip-vit-large-patch14-336`;
- revision: `ce19dc912ca5cd21c8a653c79e251e808ccabcd1`;
- layer request: `-2`, resolved to layer 22 for the validated model;
- VisionZip preset: `configs/visionzip_64.json`;
- nominal budget: 54 dominant + 10 contextual tokens;
- actual stored sequence: 65 tokens including CLS;
- hidden width: 1,024;
- storage dtype: float32;
- feature shard size: 256 samples;
- expected feature shards: 32.

Each NPZ stores:

- `sample_ids`;
- `compressed_tokens`;
- `selected_indices`;
- `assignments`.

Every shard and the sample-ID order have SHA256 entries in `outputs/phase4b/commoncatalog_cc_by_8k/features/manifest.json`. Existing complete shards are verified and reused, allowing a disconnected precompute to resume at the next shard.

### AutoDL/Jupyter Terminal — precompute in tmux

```bash
tmux new -s phase4b-features
cd /root/autodl-tmp/VisionZip-Jittor
source /etc/network_turbo
export HF_HUB_DISABLE_XET=1
export HF_HUB_ETAG_TIMEOUT=60
export HF_HUB_DOWNLOAD_TIMEOUT=600
export HF_HOME=/root/autodl-tmp/cache/huggingface
export USE_TORCH=1 USE_TF=0 USE_FLAX=0 OMP_NUM_THREADS=8
set -o pipefail
/root/miniconda3/bin/python scripts/precompute_phase4b_features.py \
  --config configs/phase4b_commoncatalog_cc_by_8k.json \
  --dataset-manifest datasets/phase4b/commoncatalog_cc_by_8k/manifest.json \
  --output-dir outputs/phase4b/commoncatalog_cc_by_8k/features \
  --cache-dir /root/autodl-tmp/cache/huggingface \
  --device cuda \
  2>&1 | tee logs/phase4b/precompute_features.log
```

## 6. Training schedule fixed before the run

The initial budget-64 pilot schedule is:

| Setting | Value |
|---|---:|
| Train samples | 7,168 |
| Held-out samples | 1,024 |
| Micro-batch | 4 |
| Gradient accumulation | 4 |
| Effective batch | 16 |
| Optimizer steps per epoch | 448 |
| Planned epochs | 3 |
| Maximum optimizer steps | 1,344 |
| Learning rate | 1e-4 |
| Warm-up | 67 optimizer steps |
| Checkpoint/evaluation interval | 112 optimizer steps |
| Retained rolling checkpoints | 4, plus best and final |
| Seed | 2026 |

Training must preserve Phase 4A invariants:

- native Jittor Projector is trainable;
- native Jittor GPT-2 is frozen and hash-identical before/after;
- optimizer parameter scope exactly equals Projector parameters;
- every loss, gradient, and update is finite;
- checkpoint resume restores Projector, Adam, scheduler, epoch/order cursor, gradient-accumulation cursor, and RNG state.

The Phase 4B trainer and evaluator are the next implementation slice after dataset/feature preparation infrastructure.

## 7. Held-out evaluation policy

### 7.1 Primary metrics

Primary acceptance metrics use all 1,024 held-out pairs:

1. target-only teacher-forced negative log-likelihood;
2. target-only perplexity derived from that NLL.

The final checkpoint must improve held-out NLL over the untrained Projector baseline measured with the same features, tokenizer, prompt, batching, and seed.

### 7.2 Secondary generated-caption metrics

A fixed 128-sample held-out subset will be generated for:

- BLEU-1;
- BLEU-4 with explicit smoothing;
- ROUGE-L.

These are single-reference metrics against one synthetic BLIP-2 caption. They will be reported with the exact tokenizer/normalization implementation and are not directly comparable to multi-reference COCO caption benchmarks.

CIDEr and SPICE are not pilot acceptance gates because the selected source supplies one synthetic reference per image and SPICE adds an external Java/parser dependency. They may be added in a later benchmark phase if a compatible multi-reference licensed evaluation set is selected.

### 7.3 Claim boundary

A successful pilot may claim:

- real licensed paired-data training executed;
- held-out loss/perplexity was measured reproducibly;
- exact generated-caption metrics were recorded;
- Projector-only optimization and resume invariants held.

It must not claim human-level caption quality, state-of-the-art performance, or direct COCO parity.

## 8. Acceptance criteria

### Dataset preparation

- preflight passes without downloading by default;
- exact pinned revision and source objects are used;
- exactly 8,192 unique samples are materialized;
- exactly 7,168 train and 1,024 held-out rows are recorded;
- all sample image hashes, licenses, source pages, creators, and source coordinates are present;
- `samples.jsonl` SHA256 matches `manifest.json`;
- rerun refuses to overwrite a completed generated dataset silently.

### Feature precompute

- exactly 8,192 sample IDs appear once and in prepared-manifest order;
- all compressed tokens are finite and shape `[65, 1024]`;
- 32 feature shards are written for shard size 256;
- all shard SHA256 values verify;
- the producer records PyTorch/Transformers versions, model revision, layer, VisionZip config, and storage dtype;
- interrupted precompute reuses only shards whose sample-ID order and arrays validate.

### Training/evaluation

- baseline, periodic, best, and final held-out metrics are recorded;
- final held-out NLL is lower than baseline;
- GPT-2 remains frozen and byte/hash unchanged;
- optimizer scope is exactly Projector-only;
- all updates are finite;
- an interrupted run resumes to the same next-batch order and optimizer/scheduler state;
- evidence summary records throughput and peak GPU memory after warm-up.

## 9. Evidence and completion boundary

Phase 4B is complete only after a fresh CUDA run and an explicit resume run both pass, followed by a Windows evidence archive containing at least:

- committed config and this plan;
- prepared dataset manifest and attribution JSONL (images excluded unless explicitly needed);
- feature manifest and shard hashes (large feature shards excluded from the normal archive unless separately requested);
- training/evaluation JSON and JSONL logs;
- checkpoint manifests/hashes;
- environment/version capture;
- test output;
- archive SHA256.

Until those artifacts exist, documentation must say **Phase 4B in progress**, not complete.
