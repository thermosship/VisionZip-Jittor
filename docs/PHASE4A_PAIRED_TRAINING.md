# Phase 4A: Real Paired Image-Text Projector Training Infrastructure

## 1. Status and scope

Formal Phase 4A infrastructure validation completed on **2026-08-03** using an
NVIDIA GeForce RTX 4090 and Jittor 1.3.11.0. The top-level report records
`passed: true`.

Phase 4A advances the Phase 3B one-step gradient smoke into a repeatable paired
image-caption training workflow:

```text
precomputed real CLIP/VisionZip tokens
              |
              v
trainable native Jittor Projector (1024 -> 768 -> 768)
              |
              v
frozen native Jittor GPT-2 small (124,439,808 parameters)
              |
              v
teacher-forced target-only causal loss
              |
              v
Projector-only Adam updates + metrics + checkpoints/resume
```

The initial dataset is intentionally tiny: three deterministic, project-generated
sample images paired with short English captions. It validates infrastructure,
finite optimization, overfitting behavior, and checkpoint recovery without
introducing a large external dataset or an unresolved data license. It is **not**
a downstream captioning-quality result.

## 2. Versioned inputs

### 2.1 Runtime configuration

`configs/phase4a_tiny_overfit.json` records:

- GPT-2 model identity and VisionZip budget (`64`, plus CLS);
- Projector architecture and learning rate;
- deterministic seed and train/validation fraction;
- shared instruction prompt and generation prefix;
- batch size, total steps, caption/generation limits;
- log/checkpoint intervals;
- CUDA resume numerical tolerance.

The validated run uses budget 64, batch size 2, 30 total steps, learning rate
`1e-4`, seed 2026, and a deterministic two-sample train / one-sample validation
split.

### 2.2 Paired manifest

`manifests/phase4a_tiny_pairs.json` uses the schema
`phase4a_paired_manifest_v1`. Every sample has a stable ID, the image filename
used by the Phase 2 real-CLIP reference, and a caption:

```json
{
  "id": "dense",
  "image_name": "dense.png",
  "caption": " dense geometric pattern"
}
```

The manifest also records a description and license statement. Duplicate sample
IDs or image names, unknown fields, empty captions, missing Phase 2 image rows,
and unsupported artifact types are rejected.

## 3. Deterministic data and supervision path

`visionzip_jittor/phase4_data.py` provides pure-Python/NumPy helpers for:

1. strict config and manifest parsing;
2. stable seed-controlled train/validation splitting;
3. deterministic epoch-shuffled mini-batch selection from a global step;
4. mapping manifest image names to precomputed Phase 2 `compressed_tokens` rows;
5. prompt/caption tokenization with an explicit EOS token and caption limit;
6. target-only labels, label masks, and attention masks.

The first run reuses the exact Phase 2 budget-64 compressed tokens rather than
re-running CLIP for every Projector step. The selected visual tensor shape is
`[3, 65, 1024]`.

The causal loss mask is zero over prompt and visual positions. Only caption and
EOS positions contribute to `masked_causal_language_loss`.

## 4. Native Jittor trainer

`scripts/run_phase4a_training.py` performs these checks and actions:

- verifies every exported GPT-2 artifact checksum;
- loads the local tokenizer and native Jittor GPT-2 weights;
- freezes all GPT-2 parameters and records before/after SHA256 values;
- creates the four-tensor `mlp2x_gelu` Projector;
- constructs Adam from Projector parameters only;
- logs step loss, full-train loss, validation loss, gradient norms, parameter
  deltas, batch IDs, and elapsed time to JSONL;
- writes periodic and final Projector/Adam checkpoints;
- validates a real `--resume` execution path;
- performs deterministic greedy decoding for the validation sample;
- writes a top-level machine-readable summary.

Generated run artifacts are ignored by Git under `outputs/phase4a/` and
`logs/phase4a/`.

## 5. Complete checkpoint format

`visionzip_jittor/phase4_training.py` defines
`phase4a_projector_checkpoint_v1`. A checkpoint contains:

- every named Projector tensor;
- every Jittor Adam first-moment and second-moment tensor;
- Adam step, learning rate, epsilon, betas, and weight decay;
- global training step;
- full Phase 4A config snapshot;
- train/validation sample IDs;
- budget and Phase 2 reference path;
- Projector and optimizer-state SHA256 values.

Checkpoint loading rejects missing/unexpected Projector keys, tensor shape
mismatches, unsupported artifact types, Projector hash mismatches, and Adam-state
hash mismatches.

The CUDA next-step check is numerically reproduced at `atol=1e-5`. It is not
bitwise exact: the validated run observed loss absolute error
`4.76837158203125e-07` and Projector maximum absolute error
`3.725290298461914e-09`. The report exposes both `bit_exact: false` and
`numerically_reproduced: true`; documentation must not rewrite this as bitwise
CUDA determinism.

## 6. Commands

### 6.1 Tests

```bash
cd /root/autodl-tmp/VisionZip-Jittor
export USE_TORCH=0 USE_TF=0 USE_FLAX=0 OMP_NUM_THREADS=8
/root/autodl-tmp/envs/visionzip-jittor/bin/python \
  -m unittest discover -s tests -v
```

### 6.2 Fresh tiny paired run

```bash
cd /root/autodl-tmp/VisionZip-Jittor
mkdir -p logs/phase4a
export USE_TORCH=0 USE_TF=0 USE_FLAX=0 OMP_NUM_THREADS=8
set -o pipefail

/root/autodl-tmp/envs/visionzip-jittor/bin/python \
  scripts/run_phase4a_training.py \
  --device cuda \
  2>&1 | tee logs/phase4a/full_console.log
```

### 6.3 Resume from step 10

```bash
/root/autodl-tmp/envs/visionzip-jittor/bin/python \
  scripts/run_phase4a_training.py \
  --device cuda \
  --resume outputs/phase4a/tiny_overfit/checkpoints/projector_step_000010.npz \
  --output-dir outputs/phase4a/resume_test \
  --log-dir logs/phase4a_resume_test
```

## 7. Validated results

### 7.1 Test suites

| Environment | Result |
|---|---|
| Windows/PyTorch development environment | 41 tests, `OK`, 13 skipped because Jittor is unavailable |
| AutoDL/Jittor environment | 41 tests, `OK`, 8 skipped because PyTorch is unavailable |

The AutoDL run includes the new checkpoint round-trip and tiny teacher-forced
Projector-only update tests.

### 7.2 Fresh 30-step run

| Field | Result |
|---|---:|
| Top-level status | `passed: true` |
| Train samples | `dense`, `scene` |
| Validation sample | `text` |
| Initial full-train loss | `9.825726509094238` |
| Final full-train loss | `4.479739189147949` |
| Train-loss improvement | `5.345987319946289` |
| Initial validation loss | `7.5414838790893555` |
| Final validation loss | `7.752180099487305` |
| All updates finite/nonzero | `true` |
| Optimizer scope Projector-only | `true` |
| GPT-2 frozen and unchanged | `true` |
| Checkpoint next step numerically reproduced | `true` |

The train loss decreases strongly, which proves that the Projector can be
optimized through the frozen native GPT-2. Validation loss does **not** improve
in the fresh run, which is expected to be unstable for a one-example validation
set and must not be presented as generalization.

### 7.3 Explicit resume run

A second invocation resumed from `projector_step_000010.npz`, continued to step
30, and also produced `passed: true`:

- `start_step: 10`;
- final train loss `4.253312587738037`;
- all updates finite;
- GPT-2 unchanged;
- checkpoint next-step numerical reproduction passed.

### 7.4 Artifact hashes

```text
final checkpoint:
  SHA256 6b3e200aa4d320b1251fced1b4a6d8fad6fad3b205ea94812a743b83c709b66b

fresh summary JSON:
  SHA256 eaf2eb9210c8ed768bfc0a4e4cbf3cf576430bedfd332f394ef417d783c7f81e

fresh metrics JSONL:
  SHA256 e2907f82015e8d6d9784a9212ac69e43e68c493d39866bc9dcbfd7ce7031d76f
```

Evidence archive transferred to Windows:

```text
File: VisionZip-Jittor-phase4a-evidence-20260803.tar.gz
SHA256: 01942F1FD7E82FAF6EB5E8BCB9FFCA9C2474B50718EEA47E00BA446960926858
Entries: 12
Windows path: C:\Users\69444\Desktop\cmm\VisionZip-Jittor-phase4a-evidence-20260803.tar.gz
```

## 8. Interpretation and non-claims

Phase 4A proves:

- a versioned real image-caption manifest can be joined to Phase 2 visual rows;
- teacher-forced target masking is deterministic;
- repeated Projector-only optimization through frozen GPT-2 is finite;
- training metrics and complete Projector/Adam checkpoints are produced;
- checkpoint loading restores hashed state and resume can continue training;
- the tiny training set can be overfit.

Phase 4A does **not** prove:

- useful image captioning or VQA quality;
- generalization from three generated images;
- convergence on a real benchmark dataset;
- that the displayed generated text is semantically grounded;
- bitwise deterministic CUDA replay;
- production throughput, mixed precision, KV-cache generation, or a larger LLM.

The validation generation is intentionally retained as execution evidence, but
its text is poor. This is consistent with training only two captions for 30
steps and is not a failure of the infrastructure acceptance criteria.

## 9. Next stage

Phase 4B should select and document a licensed paired dataset, precompute its
CLIP/VisionZip features, train for a meaningful schedule, and evaluate held-out
caption quality with reproducible metrics. Dataset license, storage, filtering,
caption length, primary token budget, batch/accumulation plan, checkpoint
retention, and validation metrics must be fixed before a large run.
