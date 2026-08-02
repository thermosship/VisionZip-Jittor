# Phase 3: Projector and Frozen Language Module Integration

## 1. Goal

Phase 3 is split into two explicitly separated parts:

- **Phase 3A (implemented here):** run real Phase 2 CLIP features through native Jittor VisionZip, a native Jittor Projector, a frozen language surrogate, a scalar loss, and backward propagation. This validates token plumbing, shapes, freezing, and gradient isolation.
- **Phase 3B (future):** replace the surrogate with a real frozen language model, then validate real generation, prefill latency, and peak memory.

The Phase 3A language component is deliberately named `FrozenLanguageStub`. It contains only a frozen token embedding and a frozen output head. It is **not a real LLM** and cannot support claims about generation quality or end-to-end LLM speedup.

## 2. Minimal data path

```text
Phase 2 real CLIP NPZ
  hidden_states + attentions + metric
              |
              v
native Jittor VisionZip
  [B, 577, 1024] -> [B, 65/129/193, 1024]
              |
              v
native Jittor MultimodalProjector
  Linear(1024, 4096) -> GELU -> Linear(4096, 4096)
              |
              v
prefix embeddings + projected visual embeddings + suffix embeddings
              |
              v
frozen language output head -> deterministic scalar loss -> backward
```

The default `mlp2x_gelu` projector is:

```text
1024 -> 4096 -> GELU -> 4096
```

A single-layer `linear` projector is also available for ablation and small tests.

## 3. Phase 3A acceptance boundary

Phase 3A must prove all of the following:

1. The 64/128/192 real-CLIP NPZ inputs are passed through native Jittor VisionZip again; the runner does not simply load the saved compressed-token array.
2. VisionZip output shapes are `[3,65,1024]`, `[3,129,1024]`, and `[3,193,1024]`.
3. The Projector output hidden size is 4096.
4. Prefix text, projected visual tokens, and suffix text are packed in the expected order.
5. CLIP/reference inputs and all language-stub parameters are stop-grad.
6. The optimizer receives Projector parameters only.
7. Every Projector parameter tensor receives a finite gradient and the total gradient norm is nonzero.
8. One Adam step changes Projector parameters while frozen language parameters remain bitwise unchanged.
9. Recomputed VisionZip outputs still pass the Phase 2 `atol=1e-5, rtol=1e-5` regression check and assignments remain exactly equal.

Phase 3A does **not** prove:

- real LLM text generation;
- downstream VLM task quality;
- real LLM prefill speedup or memory reduction;
- feasibility of training a 7B-scale model.

## 4. Added files

```text
visionzip_jittor/projector_config.py   # Projector/stub configuration
visionzip_jittor/projector.py          # linear and mlp2x_gelu Projectors
visionzip_jittor/multimodal.py         # frozen stub and embedding packing
configs/phase3_projector_smoke.json    # default 1024 -> 4096 setup
scripts/run_phase3_projector_smoke.py  # 64/128/192 forward/backward runner
tests/test_projector_config.py         # pure-Python config tests
tests/test_phase3_jittor.py            # native Jittor gradient/freeze tests
```

## 5. Run on AutoDL

Run these commands in the **AutoDL/Jupyter terminal**:

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/envs/visionzip-jittor
export OMP_NUM_THREADS=8
cd /root/autodl-tmp/VisionZip-Jittor
```

Run the full test suite first:

```bash
python -m unittest discover -s tests -v
```

Confirm that the three Phase 2 references are still present:

```bash
ls -lh \
  outputs/real_clip/reference_clip_64_code_exact_float32_real_clip.npz \
  outputs/real_clip/reference_clip_128_code_exact_float32_real_clip.npz \
  outputs/real_clip/reference_clip_192_code_exact_float32_real_clip.npz
```

Run Phase 3A:

```bash
mkdir -p logs/phase3
set -o pipefail

python scripts/run_phase3_projector_smoke.py \
  --projector-config configs/phase3_projector_smoke.json \
  --reference-dir outputs/real_clip \
  --device cuda \
  --output-json logs/phase3/projector_smoke.json \
  2>&1 | tee logs/phase3/projector_smoke.log
```

Inspect the summary:

```bash
cat logs/phase3/projector_smoke.json

grep -E '"budget"|"projector_changed"|"frozen_language_changed"|"passed"' \
  logs/phase3/projector_smoke.json
```

## 6. Passing criteria

The top-level report must contain:

```json
{
  "artifact_type": "phase3_projector_frozen_language_stub_smoke_v1",
  "real_llm": false,
  "passed": true
}
```

Each budget must satisfy:

```text
all shape_checks == true
phase2_regression.compressed_allclose == true
phase2_regression.assignments_exact == true
reference_inputs_all_stop_grad == true
frozen_language_all_stop_grad == true
gradient.finite == true
gradient.l2_norm > 0
projector_changed == true
frozen_language_changed == false
passed == true
```

## 7. Formal AutoDL result (2026-08-02)

Environment reported by the runner:

```text
Jittor: 1.3.11.0
device: cuda
Projector: mlp2x_gelu, 1024 -> 4096 -> 4096
Projector parameters: 20,979,712
Frozen-language parameters: 1,048,576
optimizer: Adam, Projector parameters only
```

All three real-CLIP budgets passed:

| Budget | VisionZip output | Projector output | Packed shape | Logits shape | Loss | Gradient L2 | Projector max delta | Frozen max delta |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | `[3,65,1024]` | `[3,65,4096]` | `[3,81,4096]` | `[3,81,128]` | `0.04205053` | `0.71024593` | `9.99989e-05` | `0.0` |
| 128 | `[3,129,1024]` | `[3,129,4096]` | `[3,145,4096]` | `[3,145,128]` | `0.03326600` | `0.48738245` | `9.99980e-05` | `0.0` |
| 192 | `[3,193,1024]` | `[3,193,4096]` | `[3,209,4096]` | `[3,209,128]` | `0.03013558` | `0.43798293` | `9.99961e-05` | `0.0` |

For each budget:

```text
shape_checks: all true
compressed_allclose: true
assignments_exact: true
optimizer_scope_exact: true
projector_all_trainable: true
parameter_tensors_with_grad: 4
gradient.finite: true
projector_changed: true
frozen_language_all_stop_grad: true
frozen_language_changed: false
reference_inputs_all_stop_grad: true
passed: true
```

The top-level result was:

```json
{
  "artifact_type": "phase3_projector_frozen_language_stub_smoke_v1",
  "real_llm": false,
  "passed": true
}
```

### Timing interpretation

The 64-budget forward/backward values (`4318 ms` / `7317 ms`) include first-use JIT or graph compilation that was not covered by the earlier forward-only warm-up. They must not be compared with the later values or reported as steady-state performance.

The 128 and 192 smoke calls reported approximately `4.57/2.56 ms` and `5.33/0.99 ms` for forward/backward. These are single smoke measurements, not a benchmark. Phase 3 performance claims require a separate runner with compilation isolation, repeated warm-up, synchronized repeated iterations, and peak-memory collection.

### Conclusion

Phase 3A is complete. It proves that real CLIP features can flow through native Jittor VisionZip and a trainable 20.98M-parameter Projector, while the reference inputs and language surrogate remain frozen. It also proves one-step backward/update mechanics and exact gradient isolation for all three budgets.

It does not prove real LLM generation or end-to-end VLM performance; those remain Phase 3B work.

## 8. Next step: Phase 3B

Now that Phase 3A has passed on AutoDL:

1. Select a license-compatible and obtainable language model with hidden size 4096.
2. Implement or integrate its embedding, transformer, LM head, and weight loading in Jittor.
3. Keep CLIP and the language model frozen; optimize the Projector only.
4. Validate generation with real prompts.
5. Record output text, prefill latency, peak memory, and loss for 64/128/192.
6. Only then decide whether to proceed to Projector fine-tuning and full evaluation.
