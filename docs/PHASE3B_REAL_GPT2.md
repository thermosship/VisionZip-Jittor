# Phase 3B: Native Jittor Real Frozen GPT-2 Integration

## 1. Status and goal

Phase 3B replaces the Phase 3A `FrozenLanguageStub` with a real autoregressive
language model. The first minimum integration targets **GPT-2 small (124M)**
because it is a real decoder-only Transformer, fits comfortably on an RTX 4090,
and is small enough that every Transformer block can be reproduced natively in
Jittor before attempting a multi-billion-parameter model.

Formal validation status on **2026-08-03**: **passed**.

- native Jittor GPT-2 embedding, learned positions, causal multi-head attention,
  MLP, residual blocks, final LayerNorm, and tied LM head: implemented and run;
- Hugging Face GPT-2 float32 artifact exporter and deterministic logit reference:
  exported and checksum-verified;
- real tokenizer loading, multimodal embedding packing, target loss, backward,
  Projector-only Adam update, greedy text generation, warmed prefill timing, and
  sampled process GPU memory: executed on AutoDL CUDA;
- five native Jittor GPT-2 unit tests: passed;
- 64/128/192 integration results: all passed;
- final report: `real_llm: true` and top-level `passed: true`.

This completes the minimum Phase 3B real-LLM integration. It does not claim a
trained visual-language Projector, downstream task quality, optimized serving,
or parity with a multi-billion-parameter VLM.

## 2. Why the Projector changes from 4096 to 768

Phase 3A used a synthetic 4096-dimensional frozen language surrogate. GPT-2
small has language hidden size 768, so Phase 3B constructs the Projector as:

```text
mlp2x_gelu: Linear(1024, 768) -> GELU -> Linear(768, 768)
```

This is an intentional model-dependent interface change, not a regression.
The visual input remains the real CLIP-L/14-336 output with hidden size 1024.

## 3. Real Phase 3B data path

```text
Phase 2 real CLIP NPZ
  hidden_states + attentions + metric
              |
              v
native Jittor VisionZip
  [B,577,1024] -> [B,65/129/193,1024]
              |
              v
trainable native Jittor Projector
  [B,N,1024] -> [B,N,768]
              |
              v
real GPT-2 token embeddings + visual embeddings + target embeddings
              |
              v
12 native Jittor GPT-2 Transformer blocks
              |
              v
real tied GPT-2 LM head -> causal language loss -> backward
```

The GPT-2 parameters and Phase 2 CLIP inputs are stop-grad. The optimizer owns
Projector parameters only. Gradients still propagate through the frozen GPT-2
operations into the Projector activations.

## 4. Added files

```text
visionzip_jittor/gpt2_config.py              # GPT-2 and Phase 3B configs
visionzip_jittor/gpt2.py                     # native Jittor GPT-2 implementation
configs/phase3b_gpt2.json                    # prompts, targets and benchmark setup
requirements/phase3b_reference.txt           # PyTorch artifact-export environment
requirements/phase3b_jittor.txt              # tokenizer deps for Jittor runtime
scripts/export_gpt2_jittor_artifacts.py       # HF -> NPZ/tokenizer/reference export
scripts/run_phase3b_gpt2.py                   # real 64/128/192 integration runner
tests/test_gpt2_config.py                     # pure-Python config tests
tests/test_gpt2_jittor.py                     # small native Jittor GPT-2 tests
```

## 5. Acceptance criteria

A formal Phase 3B pass requires all of the following:

1. Top-level `real_llm` is `true`.
2. Hugging Face GPT-2 weights are exported in float32; weight, config,
   tokenizer-tree, and reference checksums are exact; every expected native
   Jittor tensor is loaded with an exact shape match.
3. Native parameter count and loaded tensor count match the Hugging Face artifact
   manifest, which also identifies `GPT2LMHeadModel` and the configured model.
4. Native Jittor text-only logits match the deterministic Hugging Face reference
   within the configured tolerance.
5. Real GPT-2 token embeddings, 12 Transformer blocks, final LayerNorm, and tied
   LM head execute in Jittor.
6. Real Phase 2 inputs are recompressed by native Jittor VisionZip; compressed
   tokens remain allclose and assignments remain exact.
7. 64/128/192 visual outputs have 65/129/193 tokens and Projector hidden size 768.
8. The optimizer contains Projector parameters only.
9. Every Projector parameter tensor receives finite gradient and total gradient
   norm is nonzero.
10. One optimizer step changes Projector parameters.
11. Every GPT-2 parameter is stop-grad and a full before/after SHA256 over all
    language tensors is unchanged.
12. Greedy decoding returns token IDs and decoded text through the real GPT-2 LM
    head for each of the three sample images.
13. Prefill latency is measured only after configured warm-up iterations, and
    sampled process GPU memory is recorded when `nvidia-smi` is available.

## 6. AutoDL validation results (2026-08-03)

### 6.1 Environment and artifact integrity

The formal run used:

```text
GPU: NVIDIA GeForce RTX 4090, 24564 MiB
Driver: 580.105.08
Python: 3.10.20
Jittor: 1.3.11.0
PyTorch exporter: 2.1.2+cu118
Transformers exporter: 4.31.0
Model: openai-community/gpt2
Architecture: GPT2LMHeadModel
Language parameters: 124,439,808
Exported tensors: 148
Validation Git commit: c1b824490fd75a5ffbc13f3a613b56a99c12d6ca
```

All manifest checks passed:

```text
architecture_is_gpt2_lm_head: true
configured_model_matches_manifest: true
loaded_tensor_count_matches_manifest: true
parameter_count_matches_manifest: true
language_all_stop_grad: true
language_unchanged: true
```

The exported artifact hashes were:

| Artifact | SHA256 |
|---|---|
| `gpt2_float32_weights.npz` | `e6da2a75ba1c4f47274e9f635a8e3cee19efd74a30009a4f4220509bbe431124` |
| `hf_config.json` | `02fad0c9969afcce718dea8ac044a4fce769c91a071d5571c124f3ae9a546c66` |
| tokenizer tree | `bdc7acfc646e49d06bf6d8635d1e06b0cb327345e7998a60e1dc927db545149c` |
| `text_reference.npz` | `699c1e2f08f061c2be933c40b4316df362a27a798341ab08b0143767b0885d13` |

### 6.2 Native GPT-2 unit tests and text-logit alignment

All five targeted native Jittor tests passed, including Transformer-block
parameter registration, tied-head forward shape, the `inputs_embeds` path,
finite masked language loss, and full language-parameter freezing:

```text
Ran 5 tests in 0.155s
OK
```

For the deterministic prompt `A native Jittor language model`, the native
Jittor logits had shape `[1,7,50257]` and matched the Hugging Face reference:

```text
max_abs_error: 0.000213623046875
atol: 0.0005
rtol: 0.0005
allclose: true
```

### 6.3 Three-budget real-LLM results

| Budget | VisionZip output | Projector output | Packed embeddings | Logits | Loss | Compressed max abs | Gradient L2 | Prefill mean / median ms | Peak GPU MiB | Result |
|---:|---|---|---|---|---:|---:|---:|---:|---:|---|
| 64 | `[3,65,1024]` | `[3,65,768]` | `[3,78,768]` | `[3,78,50257]` | `9.202743` | `5.72205e-06` | `385.6162` | `13.4518 / 13.4294` | 1550 | PASS |
| 128 | `[3,129,1024]` | `[3,129,768]` | `[3,142,768]` | `[3,142,50257]` | `8.336858` | `1.90735e-06` | `372.3646` | `8.6873 / 8.6604` | 1896 | PASS |
| 192 | `[3,193,1024]` | `[3,193,768]` | `[3,206,768]` | `[3,206,50257]` | `8.949780` | `1.90735e-06` | `153.1143` | `10.3456 / 10.2675` | 2350 | PASS |

For every budget:

- compressed tokens were allclose to the Phase 2 reference;
- contextual assignments were exact;
- all output-shape checks passed;
- the optimizer contained the 1,377,792 Projector parameters only;
- all four Projector parameter tensors were trainable and received finite,
  nonzero gradients;
- the maximum one-step Projector update was approximately `1.00002e-04`;
- all GPT-2 tensors remained stop-grad and the complete before/after language
  SHA256 was identical;
- real GPT-2 greedy decoding produced token IDs and tokenizer-decoded strings;
- the per-budget result was `passed: true`.

The report therefore ended with:

```json
{
  "real_llm": true,
  "passed": true
}
```

### 6.4 Evidence archive

The result package was transferred from AutoDL and verified on Windows:

```text
File: VisionZip-Jittor-phase3b-evidence-20260803.tar.gz
SHA256: 27c24946ff9aaf361a0d6a16c64f66f9fd587a8537e086badb3dc15c0819859e
```

The package contains the export log, targeted unit-test log, budget-64 and full
three-budget JSON/log reports, environment and Git metadata, artifact hashes,
GPT-2 config/manifest, deterministic text reference, and the pre-result Phase 3B
documentation. The 475 MiB GPT-2 weight NPZ is intentionally excluded; its
SHA256 is preserved.

### 6.5 Timing and generation boundaries

The prefill values above use two warm-up iterations and five measured iterations.
They are useful smoke observations, but the non-monotonic values and the observed
single backward timings (`127.67`, `3492.00`, and `7.37` ms) must not be
presented as a synchronized production benchmark. Jittor lazy execution,
compilation, and cache placement can move work across timing boundaries.

The decoded strings are also not task-quality evidence. The Projector was
randomly initialized and received one optimizer step only. Generation proves
that the real frozen GPT-2 path and tokenizer execute; it does not establish
captioning or VQA quality.

## 7. Prepare artifacts in the PyTorch environment

Run in the **AutoDL/Jupyter terminal**:

```bash
source /etc/network_turbo
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
cd /root/autodl-tmp/VisionZip-Jittor

export HF_HUB_DISABLE_XET=1
export HF_HUB_ETAG_TIMEOUT=60
export HF_HUB_DOWNLOAD_TIMEOUT=600
export HF_HOME=/root/autodl-tmp/cache/huggingface
export USE_TORCH=1
export USE_TF=0
export USE_FLAX=0
mkdir -p logs/phase3b

/root/miniconda3/bin/python \
  scripts/export_gpt2_jittor_artifacts.py \
  --model-name-or-path openai-community/gpt2 \
  --cache-dir /root/autodl-tmp/cache/huggingface \
  --output-dir outputs/phase3b/gpt2 \
  2>&1 | tee logs/phase3b/export_gpt2.log
```

Expected artifacts:

```text
outputs/phase3b/gpt2/gpt2_float32_weights.npz
outputs/phase3b/gpt2/hf_config.json
outputs/phase3b/gpt2/manifest.json
outputs/phase3b/gpt2/text_reference.npz
outputs/phase3b/gpt2/tokenizer/
```

The weight NPZ is large and intentionally ignored by Git. Conda activation does
not clear `USE_TORCH`; if the shell was previously used for the Jittor runner,
`USE_TORCH=0` would make Transformers deliberately hide an otherwise installed
PyTorch. The exporter now forces `USE_TORCH=1`, and the commands above state it
explicitly as well.

## 8. Prepare the Jittor environment

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/envs/visionzip-jittor
cd /root/autodl-tmp/VisionZip-Jittor

python -m pip install -r requirements/phase3b_jittor.txt
export OMP_NUM_THREADS=8
export USE_TORCH=0
export USE_TF=0
export USE_FLAX=0
export NVIDIA_TF32_OVERRIDE=0
```

Run tests first:

```bash
python -m unittest discover -s tests -v
```

## 9. Execute incrementally

Start with budget 64 so native operator/API problems can be corrected before a
full three-budget run:

```bash
mkdir -p logs/phase3b
set -o pipefail

python scripts/run_phase3b_gpt2.py \
  --config configs/phase3b_gpt2.json \
  --artifact-dir outputs/phase3b/gpt2 \
  --reference-dir outputs/real_clip \
  --budgets 64 \
  --device cuda \
  --output-json logs/phase3b/gpt2_smoke_64.json \
  2>&1 | tee logs/phase3b/gpt2_smoke_64.log
```

After budget 64 passes, run all budgets:

```bash
python scripts/run_phase3b_gpt2.py \
  --config configs/phase3b_gpt2.json \
  --artifact-dir outputs/phase3b/gpt2 \
  --reference-dir outputs/real_clip \
  --device cuda \
  --output-json logs/phase3b/gpt2_smoke.json \
  2>&1 | tee logs/phase3b/gpt2_smoke.log
```

Inspect critical fields:

```bash
grep -E \
  '"real_llm"|"allclose"|"compressed_allclose"|"assignments_exact"|"optimizer_scope_exact"|"projector_changed"|"language_all_stop_grad"|"language_unchanged"|"passed"' \
  logs/phase3b/gpt2_smoke.json
```

## 10. Interpretation limits

The generation path is real GPT-2, but the Projector starts from random weights
and receives only one smoke-test optimizer step. Generated strings prove model
execution and decoding, **not visual-language quality**.

The current greedy decoder intentionally recomputes the full sequence and does
not implement a KV cache. Generation latency therefore must not be presented as
optimized serving speed. The reported prefill benchmark is the relevant initial
64/128/192 comparison and uses configured warm-up and repeated iterations.

This phase also does not claim parity with a 7B VLM. A larger model, trained
Projector checkpoint, downstream evaluation, mixed precision, KV cache, and
production memory optimization remain later work.
