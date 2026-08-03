# VisionZip-Jittor

使用 **Jittor 原生张量算子**复现 VisionZip 的视觉 Token 压缩核心，并建立可重复的 PyTorch/Jittor 数值对齐流程。

> Current status: **Phases 1, 2, 3A, 3B, 4A, and 4B are complete; Phase 5A formal acceptance has passed and evidence finalization is in progress.** Native Jittor GPT-2 KV-cache generation passed the clean-commit acceptance-v3 protocol at `7227717`: exact greedy IDs, exact cache contracts, and the frozen per-step total-variation bound passed all 9/9 budget/sample cases; GPT-2 and Projector remained frozen and hash-unchanged. The global maximum TV was `3.505422367609121e-05` under the frozen `5e-5` bound. Final documentation and a cross-host verified evidence archive remain before Phase 5A is marked complete.

## 1. 项目目标

本项目计划使用 Jittor 复现 VisionZip，最终覆盖：

- Dominant Token Selection；
- Contextual Token Merging；
- 与官方 PyTorch 实现的算子级对齐；
- 训练无关的多模态推理；
- 仅训练多模态 Projector 的高效微调；
- 训练、测试、性能和可视化日志。

Completed scope includes Phases 1, 2, 3A, 3B, 4A, and 4B:

- 原生 Jittor VisionZip 核心；
- 独立 PyTorch 参考实现；
- 64、128、192 Token 配置；
- PyTorch 参考张量导出；
- Jittor 中间变量逐项对齐；
- 单元测试和核心性能测试入口；
- 真实图片的 CLIP Hidden State、Attention 与 Key Metric 导出；
- 真实特征的一键 PyTorch/Jittor 对齐流水线；
- Dominant Token 与 Contextual Merge 可视化；
- 面向 CLIP 64 维 Key Metric 的 PyTorch 2.1 CUDA 精确归一化路径；
- 真实 CLIP 特征在 64/128/192 三档下的逐项对齐与 9 张 Token 可视化；
- 原生 Jittor `1024 -> 768 -> 768` Projector、真实多模态 embedding packing 与 Projector-only 优化；
- 原生 Jittor GPT-2 small（124M）12 层因果 Transformer、权重加载、冻结校验和真实解码；
- 64/128/192 三档真实 CLIP → VisionZip → Projector → GPT-2 路径的 CUDA 验收。;
- Phase 4A versioned paired manifests, deterministic splits, and target-only causal masks;
- repeated Projector-only training on precomputed real CLIP/VisionZip tokens, JSONL metrics, complete Projector/Adam checkpoints, and resume.
- Phase 4B pinned licensed-data configuration, deterministic 7,168/1,024 split, row-level attribution records, storage preflight, and 32 verified frozen-feature shards;
- 1,344-step real paired Projector-only training with exact frozen-GPT-2 invariants, checkpoint resume, full held-out NLL/perplexity, deterministic single-reference BLEU/ROUGE, and accepted post-warm-up throughput/peak-memory evidence;
- native Jittor GPT-2 per-layer KV-cache prefill/decode, cached-versus-uncached greedy correctness checks, and a reproducible Phase 5A latency/memory runner.

## 2. 上游版本与复现范围

第一阶段参考逻辑固定到：

```text
Repository: JIA-Lab-research/VisionZip
Branch: main
Commit: 8f86b55c6f000eb033e6912538af2dd7dcb30502
Snapshot date: 2026-08-01
```

详细版本记录见 [`references/UPSTREAM.md`](references/UPSTREAM.md)，第一阶段数值结果见 [`docs/PHASE1_RESULTS.md`](docs/PHASE1_RESULTS.md)，第二阶段真实 CLIP 实测结果见 [`docs/PHASE2_REAL_CLIP.md`](docs/PHASE2_REAL_CLIP.md)，Phase 3A Projector smoke 见 [`docs/PHASE3_PROJECTOR_FROZEN_LLM.md`](docs/PHASE3_PROJECTOR_FROZEN_LLM.md)，Phase 3B 真实 GPT-2 结果见 [`docs/PHASE3B_REAL_GPT2.md`](docs/PHASE3B_REAL_GPT2.md)。 Phase 4A paired-training results are in [`docs/PHASE4A_PAIRED_TRAINING.md`](docs/PHASE4A_PAIRED_TRAINING.md).

Phase 4B dataset/licensing policy is fixed in [`docs/PHASE4B_DATASET_PLAN.md`](docs/PHASE4B_DATASET_PLAN.md), and the executed training, held-out metrics, benchmark evidence, integrity checks, and claim boundary are recorded in [`docs/PHASE4B_REAL_PAIRED_TRAINING.md`](docs/PHASE4B_REAL_PAIRED_TRAINING.md). Phase 5A KV-cache scope, fixed artifacts, correctness gates, and benchmark protocol are defined in [`docs/PHASE5A_KV_CACHE_PLAN.md`](docs/PHASE5A_KV_CACHE_PLAN.md).

本仓库没有复制官方 LLaVA 模型代码。`reference/pytorch_visionzip.py` 是为了框架数值对齐而编写的独立参考模块。

## 3. 核心算法

### 3.1 Dominant Token Selection

对于带 CLS Token 的 CLIP：

1. 读取视觉编码器倒数第二层 Attention；
2. 对 `CLS -> Patch` Attention 在多头维度求和；
3. Top-k 选择 Dominant Patch；
4. 额外保留 CLS；
5. 按原视觉序列顺序输出 Dominant Tokens。

需要注意：PyTorch `topk` 返回的是按分数排序的索引，但官方代码随后使用布尔 Mask 提取 Token，所以实际输出保持原始序列顺序。Jittor 版本显式重建了这一行为。

### 3.2 Contextual Token Merging

1. 删除 CLS 和 Dominant Patch；
2. 对 Key Metric 做 L2 归一化；
3. 均匀选取 Contextual Target；
4. 用余弦相似度将其他 Token 分配给最相似的 Target；
5. 使用原生 Jittor 比较、One-hot 和 Batch Matrix Multiplication 聚合 Hidden State；
6. 拼接 Dominant Tokens 与 Contextual Tokens。

核心压缩代码不通过 Torch 兼容层执行。

### 3.3 两种合并语义

默认的 `code_exact` 严格复现官方代码：

```text
context = target + sum(assigned) / max(count, 1)
```

仓库额外提供 `paper_avg` 消融：

```text
context = (target + sum(assigned)) / (count + 1)
```

正式对齐使用 `code_exact`。详细说明见 [`docs/ALGORITHM_ALIGNMENT.md`](docs/ALGORITHM_ALIGNMENT.md)。

## 4. Token 预算口径

官方 64 Token 设置为：

```text
54 dominant patches + 10 contextual tokens = 64 nominal visual tokens
```

CLIP 路径还保留 CLS，所以实际输出为：

```text
CLS + 54 dominant patches + 10 contextual tokens = 65 tokens
```

| 配置 | Dominant Patch | Contextual | 名义预算 | 实际输出（含 CLS） |
|---|---:|---:|---:|---:|
| `visionzip_64.json` | 54 | 10 | 64 | 65 |
| `visionzip_128.json` | 108 | 20 | 128 | 129 |
| `visionzip_192.json` | 162 | 30 | 192 | 193 |

128 和 192 的拆分是本项目按 54:10 比例扩展的实验预设，不将其表述为论文官方固定拆分。

## 5. 第一阶段对齐结果

实验使用 CLIP ViT-L/14-336 路径的典型 Shape：

```text
hidden_states: [1, 577, 1024]
attentions:    [1, 16, 577, 577]
metric:        [1, 577, 64]
```

| 名义预算 | 实际输出 | 压缩 Token 最大绝对误差 | 索引一致率 | Assignment 一致率 | 结果 |
|---:|---:|---:|---:|---:|---|
| 64 | 65 | 2.3841858e-07 | 100% | 100% | PASS |
| 128 | 129 | 2.3841858e-07 | 100% | 100% | PASS |
| 192 | 193 | 2.3841858e-07 | 100% | 100% | PASS |

所有浮点张量均在 `atol=1e-5, rtol=1e-5` 下通过 `allclose`。

## 6. 已验证环境

```text
OS: Ubuntu 22.04.1 LTS
GPU: NVIDIA GeForce RTX 4090 24GB
Driver: 570.124.04
CUDA Toolkit: 11.8.89
Python: 3.10.8
PyTorch baseline: 2.1.2+cu118
Jittor: 1.3.11.0
GCC/G++: 11.3.0
Jittor CUDA arch: sm_89
```

Jittor 环境：

```text
/root/autodl-tmp/envs/visionzip-jittor
```

Jittor 缓存：

```text
/root/.cache/jittor -> /root/autodl-tmp/cache/jittor
```

## 7. AutoDL 每次开机后的操作

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/envs/visionzip-jittor
export OMP_NUM_THREADS=8
cd /root/autodl-tmp/VisionZip-Jittor
git pull
```

也可以运行：

```bash
source /root/autodl-tmp/VisionZip-Jittor/environment/activate_jittor.sh
```

## 8. 验证流程

PyTorch 与 Jittor 使用两个独立解释器：

```text
PyTorch: /root/miniconda3/bin/python
Jittor:  /root/autodl-tmp/envs/visionzip-jittor/bin/python
```

### 8.1 单元测试

```bash
python -m unittest discover -s tests -v
```

测试会根据当前环境是否安装 PyTorch/Jittor 自动跳过另一框架的测试。

### 8.2 小规模跨框架对齐

```bash
python scripts/smoke_test.py \
  --torch-python /root/miniconda3/bin/python \
  --jittor-python /root/autodl-tmp/envs/visionzip-jittor/bin/python
```

### 8.3 完整 Shape 的 PyTorch 参考导出

```bash
/root/miniconda3/bin/python scripts/export_pytorch_reference.py \
  --config configs/visionzip_64.json \
  --output outputs/reference_64_fp32_full.npz \
  --batch-size 1 \
  --sequence-length 577 \
  --hidden-dim 1024 \
  --metric-dim 64 \
  --heads 16 \
  --dtype float32
```

### 8.4 Jittor 数值对齐

```bash
/root/autodl-tmp/envs/visionzip-jittor/bin/python \
  scripts/run_jittor_alignment.py \
  --reference outputs/reference_64_fp32_full.npz \
  --output-json logs/alignment_64_fp32_full.json \
  --atol 1e-5 \
  --rtol 1e-5
```

### 8.5 Jittor 核心性能测试

```bash
python scripts/benchmark_jittor_core.py \
  --config configs/visionzip_64.json \
  --warmup 20 \
  --iterations 100 \
  --output-json logs/benchmark_core_64.json
```

性能脚本报告输入/输出 Shape、Token 压缩比例、预热后延迟、吞吐量、Jittor 版本和 GPU 信息。该结果只表示核心 Token 压缩模块，不代表完整 VLM 端到端性能。

在 RTX 4090、Batch Size 1、FP32、输入 Shape `[1, 577, 1024]`、20 次预热和 100 次正式计时下，得到：

| 名义预算 | 实际输出（含 CLS） | 序列缩减率 | 平均延迟 (ms) | 中位延迟 (ms) | 标准差 (ms) | 约合吞吐量 (calls/s) |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 65 | 88.73% | 0.977824 | 0.919659 | 0.172511 | 1022.68 |
| 128 | 129 | 77.64% | 0.924812 | 0.870302 | 0.194281 | 1081.30 |
| 192 | 193 | 66.55% | 0.943357 | 0.892080 | 0.153472 | 1059.00 |

三档核心延迟均约为 `0.9–1.0 ms`。档位之间的差值很小，且处于单次运行抖动范围内，因此不能据此宣称 128 Token 配置必然比 64 Token 配置更快。完整数据、测试边界和解释见 [`docs/PHASE1_RESULTS.md`](docs/PHASE1_RESULTS.md)。

### 8.6 真实 CLIP 特征对齐（第二阶段）

先在 PyTorch 基准环境安装额外依赖并生成样例图：

```bash
/root/miniconda3/bin/python -m pip install -r requirements/real_clip.txt
/root/miniconda3/bin/python scripts/create_sample_images.py \
  --output-dir assets/sample_images
```

将 Hugging Face 缓存放在数据盘，然后从当前 Jittor 环境一键调用两个 Python 环境：

```bash
export HF_HOME=/root/autodl-tmp/cache/huggingface
export TRANSFORMERS_CACHE=/root/autodl-tmp/cache/huggingface/transformers

python scripts/run_real_clip_pipeline.py \
  --torch-python /root/miniconda3/bin/python \
  --jittor-python /root/autodl-tmp/envs/visionzip-jittor/bin/python \
  --image-dir assets/sample_images \
  --model-name-or-path openai/clip-vit-large-patch14-336 \
  --cache-dir /root/autodl-tmp/cache/huggingface \
  --device cuda \
  --dtype float32
```

该命令提取真实图像的倒数第二层 CLIP 特征，运行 64/128/192 三档 PyTorch 参考压缩，调用原生 Jittor 对齐，并输出 Token 可视化。详细说明、输出文件和通过标准见 [`docs/PHASE2_REAL_CLIP.md`](docs/PHASE2_REAL_CLIP.md)。

### 8.7 第二阶段实测结果

2026-08-02 在 RTX 4090 上使用三张确定性样例图 `dense.png`、`scene.png` 和 `text.png` 完成正式 FP32 流水线。输入 Shape 为：

```text
hidden_states: [3, 577, 1024]
attentions:    [3, 16, 577, 577]
metric:        [3, 577, 64]
pixel_values:  [3, 3, 336, 336]
```

| 名义预算 | 实际输出 Shape | 压缩 Token 最大绝对误差 | Contextual Token 最大绝对误差 | Assignment | 结果 |
|---:|---:|---:|---:|---:|---|
| 64 | `[3, 65, 1024]` | `5.7220459e-06` | `5.7220459e-06` | exact, 100% | PASS |
| 128 | `[3, 129, 1024]` | `1.9073486e-06` | `1.9073486e-06` | exact, 100% | PASS |
| 192 | `[3, 193, 1024]` | `1.9073486e-06` | `1.9073486e-06` | exact, 100% | PASS |

所有 `selected_indices`、`dominant_ordered_indices`、`remaining_indices`、`target_positions`、`merge_positions` 和 `assignments` 均逐元素完全一致；所有规定浮点张量均在 `atol=1e-5, rtol=1e-5` 下通过。流水线最终写出 `passed: true`，并生成 `3 张图片 × 3 个预算 = 9` 张 Token 可视化。

真实 CLIP 特征包含极近的余弦相似度 tie。普通跨框架 L2 归一化曾造成 `4 / 1536` 个 Assignment 差异。`visionzip_jittor/core.py` 现通过原生 `jt.code` CUDA Kernel 复现 PyTorch 2.1 的 64 维归约布局，并使用 `__fmul_rn`、`__fadd_rn`、`__fsqrt_rn` 和 `__fdiv_rn` 固定 float32 舍入路径；修复后 norm、normalized metric、similarity 和 Assignment 的诊断结果均达到逐元素精确一致。非 CUDA、非 float32、非 64 维或 `eps > 0` 时仍使用通用 Jittor 回退路径。

### 8.8 Projector and frozen-language-stub smoke (Phase 3A)

Phase 3A uses an explicitly named `FrozenLanguageStub`. It is not a real LLM; it validates token plumbing, shapes, freezing, gradient isolation, and one Projector update.

Run this in the AutoDL environment while retaining the three Phase 2 real-CLIP NPZ files:

```bash
python scripts/run_phase3_projector_smoke.py \
  --projector-config configs/phase3_projector_smoke.json \
  --reference-dir outputs/real_clip \
  --device cuda \
  --output-json logs/phase3/projector_smoke.json \
  2>&1 | tee logs/phase3/projector_smoke.log
```

The runner recomputes native Jittor VisionZip outputs, maps 1024-dimensional tokens to 4096 dimensions, packs multimodal embeddings, checks finite nonzero Projector gradients, updates Projector parameters, and verifies that frozen language parameters do not change. See [`docs/PHASE3_PROJECTOR_FROZEN_LLM.md`](docs/PHASE3_PROJECTOR_FROZEN_LLM.md) for scope and acceptance criteria.

### 8.9 Phase 3A formal AutoDL results

On 2026-08-02, the CUDA smoke runner completed all three real-CLIP budgets with top-level `passed: true`.

| Budget | VisionZip output | Projector output | Packed embeddings | Loss | Gradient L2 norm | Projector max delta | Frozen max delta | Result |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 64 | `[3,65,1024]` | `[3,65,4096]` | `[3,81,4096]` | `0.04205053` | `0.71024593` | `9.99989e-05` | `0.0` | PASS |
| 128 | `[3,129,1024]` | `[3,129,4096]` | `[3,145,4096]` | `0.03326600` | `0.48738245` | `9.99980e-05` | `0.0` | PASS |
| 192 | `[3,193,1024]` | `[3,193,4096]` | `[3,209,4096]` | `0.03013558` | `0.43798293` | `9.99961e-05` | `0.0` | PASS |

For every budget, all shape checks passed, the Phase 2 compressed-token regression remained within tolerance, assignments were exact, the optimizer contained Projector parameters only, all four Projector parameter tensors received finite nonzero gradients, Projector parameters changed after one Adam step, and all 1,048,576 frozen-language parameters remained bitwise unchanged.

The 64-budget forward/backward values include first-use JIT/graph compilation (`4318 ms` / `7317 ms`) and are not performance measurements. The later 128/192 values are smoke timings only; a dedicated warm-up and repeated-iteration benchmark is required before making speed claims.

### 8.10 Phase 3B real frozen GPT-2 integration

Phase 3B replaces the Phase 3A stub with GPT-2 small and executes real token embeddings, learned positional embeddings, 12 causal Transformer blocks, final LayerNorm, and the tied LM head in native Jittor. The Projector maps `1024 -> 768 -> 768` and is the only optimized module.

Formal AutoDL validation completed on **2026-08-03** using an NVIDIA GeForce RTX 4090. The final report records `real_llm: true`, top-level `passed: true`, exact artifact/model integrity checks, aligned text-only logits, frozen and unchanged GPT-2 weights, and passing 64/128/192 results.

| Budget | VisionZip output | Projector output | Packed embeddings | Loss | Phase 2 max abs | Prefill mean (ms) | Peak GPU MiB | Result |
|---:|---|---|---|---:|---:|---:|---:|---|
| 64 | `[3,65,1024]` | `[3,65,768]` | `[3,78,768]` | `9.202743` | `5.72205e-06` | `13.4518` | 1550 | PASS |
| 128 | `[3,129,1024]` | `[3,129,768]` | `[3,142,768]` | `8.336858` | `1.90735e-06` | `8.6873` | 1896 | PASS |
| 192 | `[3,193,1024]` | `[3,193,768]` | `[3,206,768]` | `8.949780` | `1.90735e-06` | `10.3456` | 2350 | PASS |

The Hugging Face/Jittor text-logit maximum absolute error was `2.13623e-04` at `atol=rtol=5e-4`. For every budget, compressed tokens were allclose, assignments were exact, the optimizer scope was Projector-only, all four Projector tensors received finite nonzero gradients, Projector parameters changed, and all 124,439,808 GPT-2 parameters remained stop-grad and SHA256-identical before/after the update. Five native Jittor GPT-2 unit tests also passed.

The timings are smoke-run observations rather than a synchronized production benchmark. Generated strings prove real GPT-2 execution and tokenizer decoding only; the randomly initialized Projector received one update and has not been trained for visual-language quality. Full results and limits are documented in [`docs/PHASE3B_REAL_GPT2.md`](docs/PHASE3B_REAL_GPT2.md) Phase 4A paired-training results are in [`docs/PHASE4A_PAIRED_TRAINING.md`](docs/PHASE4A_PAIRED_TRAINING.md).

### 8.11 Phase 4A real paired Projector training

Implementation milestone: `7a62be2 feat: add phase-four paired projector training`.

Phase 4A turns the one-step Phase 3B path into a repeatable paired image-text training loop while keeping CLIP/VisionZip features and all 124,439,808 GPT-2 parameters frozen. The checked-in fixture joins the three deterministic sample images to captions, uses `dense` and `scene` for training, reserves `text` for validation, and trains only the native Jittor `1024 -> 768 -> 768` Projector at nominal budget 64 (`65` visual tokens including CLS).

The runner provides a versioned manifest, deterministic split and mini-batch order, caption truncation plus EOS, target-only teacher-forced labels, JSONL metrics, atomic checkpoints containing all Projector tensors and Adam first/second moments, strict hash/shape validation, real `--resume`, and internal next-step replay verification.

The formal RTX 4090 fresh run on **2026-08-03** completed 30 steps with top-level `passed: true`:

| Field | Result |
|---|---:|
| Initial full-train loss | `9.825726509094238` |
| Final full-train loss | `4.479739189147949` |
| Train-loss improvement | `5.345987319946289` |
| Initial validation loss | `7.5414838790893555` |
| Final validation loss | `7.752180099487305` |
| Projector-only optimizer scope | `true` |
| GPT-2 frozen and unchanged | `true` |
| All updates finite/nonzero | `true` |
| Checkpoint next step numerically reproduced within `1e-5` | `true` |

An explicit second invocation resumed from step 10 and continued to step 30 with `passed: true`. Complete checkpoint state restores hash-exactly, but the subsequent CUDA step is not claimed bitwise deterministic: observed replay errors were `4.76837e-07` for loss and `3.72529e-09` for Projector parameters, both within the declared `1e-5` tolerance.

This three-image run is an infrastructure smoke/overfit test. Its validation loss did not improve and its generated text was poor; neither is presented as captioning quality or generalization evidence. Full commands, schemas, hashes, acceptance criteria, and non-claims are in [`docs/PHASE4A_PAIRED_TRAINING.md`](docs/PHASE4A_PAIRED_TRAINING.md). Evidence archive: `VisionZip-Jittor-phase4a-evidence-20260803.tar.gz`, SHA256 `01942F1FD7E82FAF6EB5E8BCB9FFCA9C2474B50718EEA47E00BA446960926858` (12 entries).

### 8.12 Phase 4B licensed real paired training (acceptance run passed)

Phase 4B materialized exactly 8,192 CC-BY samples from the pinned `common-canvas/commoncatalog-cc-by` revision, retained row-level attribution/provenance, and produced 32 verified float32 CLIP/VisionZip feature shards at nominal budget 64 (65 tokens including CLS). The deterministic split is 7,168 train and 1,024 held-out samples.

The benchmark-instrumented RTX 4090 run completed all 1,344 optimizer steps with `passed: true`, finite Projector-only updates, and a frozen/hash-unchanged 124,439,808-parameter GPT-2. Held-out target NLL improved from `6.643716599545368` to `2.413187829110486` (63.6771% reduction), and perplexity improved from `767.9438354472138` to `11.169510943754625`.

After 67 warm-up updates, steps 68-1,344 averaged `120.37183717184918 ms` per optimizer update, `132.92145717737577` effective samples/s, and `1413.1492154945145` target tokens/s. The current Python process peaked at `3058 MiB` GPU memory across 1,416 samples.

The deterministic 128-sample generated-caption subset recorded BLEU-1 `0.28304947283049475`, add-one-smoothed BLEU-4 `0.05727683512769526`, and ROUGE-L `0.26176086973563917`. These use one BLIP-2 synthetic reference per image, are not directly comparable with multi-reference COCO metrics, and do not establish high-quality captioning. Full details are in [`docs/PHASE4B_REAL_PAIRED_TRAINING.md`](docs/PHASE4B_REAL_PAIRED_TRAINING.md). Final evidence archive: `VisionZip-Jittor-phase4b-evidence-final-v2-20260803.tar.gz`, SHA256 `2EFEEAA88F18AB11B8431A7DD810B296366073D14B5717D02C72152DBA70C032` (45 files). AutoDL and Windows hashes match; the earlier `VisionZip-Jittor-phase4b-evidence-20260803.tar.gz` remains preserved.


### 8.13 Phase 5A native Jittor GPT-2 KV-cache generation (formal acceptance passed)

Phase 5A adds a per-layer GPT-2 key/value cache with shape `[batch, heads, cached_tokens, head_dim]`, past-aware position IDs and causal masks, cache validation, and greedy generation from packed multimodal embeddings. The original full-recompute path remains the correctness oracle.

The preserved acceptance-v1 clean run at `6d3ba71` kept exact greedy IDs for all 9/9 samples and every cache/model invariant but failed its original raw-logit `allclose(1e-5, 1e-5)` gate. Acceptance v2 also remains a preserved failed dirty smoke because coordinatewise softmax allclose failed in 6/9 samples. PyTorch 2.1.2 CUDA reproduced the same class of shape-dependent FP32 drift while preserving exact IDs, so neither failed result was deleted or retroactively relabeled.

Acceptance v3 freezes the semantic gate before the formal retry: exact greedy token IDs, exact cache layer/shape/final-length checks, and per-step stable-softmax total variation no greater than `5e-5`. Raw logits, centered logits, and coordinatewise softmax allclose remain recorded as `diagnostic_only`. GPT-2 and Projector provenance, stop-grad, and SHA256 invariants remain mandatory. Speedup is reported, not required.

The clean formal run used synchronized commit `72277174069b9cef63831529f3ef2e3e16965cd1`, all three budgets, all three samples, 32 generated tokens, 3 warmups, and 10 measured runs. It recorded:

```text
tracked source tree clean:        true
exact greedy token sequences:    9 / 9
exact cache contracts:           9 / 9
TV within frozen 5e-5 bound:     9 / 9
global maximum TV:               3.505422367609121e-05
GPT-2 frozen/hash unchanged:     true / true
Projector frozen/hash unchanged: true / true
invariants_passed:               true
top-level passed:                true
```

Formal acceptance-v3 performance on the pinned RTX 4090/Jittor 1.3.11.0 environment:

| Budget | Cached prefill | Cached decode-only | Cached total | Uncached total | Speedup | Peak process GPU memory |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | `3.2958 ms` | `235.2201 ms` | `240.8540 ms` | `243.3399 ms` | `1.01032x` | `1168 MiB` |
| 128 | `3.3615 ms` | `236.9930 ms` | `241.4139 ms` | `255.4643 ms` | `1.05820x` | `1250 MiB` |
| 192 | `3.7474 ms` | `231.5706 ms` | `241.0673 ms` | `290.8731 ms` | `1.20661x` | `1322 MiB` |

The formal diagnostics intentionally still show that raw-logit allclose passed only 5/9 samples, centered-logit allclose passed 0/9, and coordinatewise probability allclose passed 4/9. Their global maxima were `0.00316619873046875`, `0.0005196525940505126`, and `2.736421851434745e-05`, respectively. These diagnostic failures do not override exact generation decisions, exact cache semantics, or the frozen distribution-level gate.

Phase 5A validates generation decisions, cache behavior, distribution-level alignment, and runtime measurements; it does **not** claim raw-logit bitwise/strict-`1e-5` equality, improved caption quality, or universal speedup. Full protocol, preserved failures, diagnosis, and claim boundaries are in [`docs/PHASE5A_KV_CACHE_PLAN.md`](docs/PHASE5A_KV_CACHE_PLAN.md).

## 9. 项目结构

```text
VisionZip-Jittor/
├── assets/sample_images/       # 可重复样例图生成位置
├── configs/                    # VisionZip, Projector, GPT-2, Phase 4A, and Phase 4B configs
├── manifests/                  # versioned paired image-text manifests
├── datasets/                   # generated Phase 4B images/manifests, not committed
├── docs/                       # 算法和实验结果说明
├── environment/                # AutoDL 激活和环境证据脚本
├── logs/                       # generated run logs, not committed
├── outputs/                    # generated weights/checkpoints/tensors, not committed
├── reference/                  # PyTorch 参考实现
├── references/                 # 上游版本记录
├── requirements/               # 两个环境的依赖说明
├── scripts/                    # 随机/真实特征导出、对齐、性能和可视化脚本
├── tests/                      # 单元测试
└── visionzip_jittor/
    |-- config.py
    |-- core.py                 # native Jittor VisionZip core
    |-- projector_config.py
    |-- projector.py            # native Jittor multimodal Projector
    |-- multimodal.py           # frozen language stub and packing
    |-- gpt2_config.py          # Phase 3B GPT-2/runtime configuration
    |-- gpt2.py                 # native Jittor real GPT-2
    |-- phase4_config.py        # Phase 4A paired-training configuration
    |-- phase4_data.py          # manifest, split, feature and mask helpers
    |-- phase4_training.py      # batches, metrics and complete checkpoints
    |-- phase4b_config.py       # pinned dataset/feature/training/evaluation plan
    |-- phase4b_data.py         # licensed sample filtering, attribution and split
    `-- phase4b_features.py     # hashed sharded frozen-feature store
```

## 10. 阶段状态

- [x] PyTorch 参考实现与固定的官方 CLIP 代码逻辑一致；
- [x] 原生 Jittor 核心实现完成；
- [x] PyTorch 侧单元测试通过；
- [x] AutoDL 上 Jittor 小张量执行测试通过；
- [x] FP32 中间索引 100% 一致；
- [x] FP32 Assignment 100% 一致；
- [x] 浮点结果在指定容差内通过；
- [x] 64/128/192 三种配置完成对齐；
- [x] 完成 RTX 4090 核心性能测试并保存摘要；
- [x] 在 AutoDL 上完成真实 CLIP 特征三档对齐和可视化；
- [x] 定位并修复真实 CLIP near-tie 的 CUDA 归一化数值路径；
- [x] 保存第二阶段流水线摘要、逐档对齐报告和 9 张可视化。

- [x] Implement the Phase 3A native Jittor Projector and multimodal packing;
- [x] Implement the frozen-language-stub gradient isolation path;
- [x] Add a 64/128/192 forward/backward smoke runner and tests;
- [x] Run Phase 3A on AutoDL RTX 4090 and save a `passed: true` report;
- [x] Implement native Jittor GPT-2 blocks, tied LM head, weight loading, and tokenizer artifacts;
- [x] Implement Phase 3B Projector-only loss/backward, real generation, prefill timing, and memory sampling;
- [x] Run the Phase 3B real GPT-2 path on AutoDL for 64/128/192 and save a `real_llm: true`, `passed: true` report;
- [x] Add a versioned Phase 4A paired manifest, deterministic split, target masks, and precomputed-feature loader;
- [x] Add repeated Projector-only training, JSONL metrics, complete Projector/Adam checkpoints, and resume;
- [x] Run the Phase 4A fresh and resumed paths on AutoDL and save `passed: true` reports;
- [x] Pin the Phase 4B CommonCatalog CC-BY revision, five pilot shards, exact 8,192-sample target, attribution policy, deterministic split, and disk estimate;
- [x] Implement Phase 4B preparation/preflight and hashed sharded feature infrastructure with unit tests;
- [x] Run Phase 4B preflight on AutoDL, materialize the licensed pilot subset, and precompute all 32 frozen feature shards;
- [x] Implement and run Phase 4B gradient-accumulated training plus held-out evaluation and benchmark instrumentation;
- [x] Define the Phase 5A KV-cache scope, fixed artifacts, tolerances, and benchmark protocol;
- [x] Implement native Jittor GPT-2 KV caching, cached greedy decoding, focused tests, and the formal runner;
- [x] Pass the reduced real-artifact Phase 5A runner smoke and full 63-test AutoDL discovery;
- [x] Commit/synchronize Phase 5A v1 and preserve the first clean formal benchmark plus numerical diagnostics;
- [x] Implement and validate acceptance v3 with a passing 9-sample dirty smoke under a new namespace;
- [x] Commit/synchronize acceptance v3 and pass clean-commit tests plus the full 64/128/192 formal protocol;
- [ ] Build and cross-host verify the Phase 5A evidence archive, record its SHA256, and mark the phase complete.

## 11. Next steps

1. Update the final Phase 5A result documentation and authoritative handoff with commit `7227717` measurements.
2. Build a versioned Phase 5A evidence archive containing the formal JSON/log/metadata, clean-test evidence, preserved v1/v2/v3 failure and diagnosis logs, reviewed source/config/tests/docs, and an internal `SHA256SUMS`.
3. Verify the archive and internal checksums on AutoDL, copy it to Windows, and confirm AutoDL/Windows SHA256 equality.
4. Record the verified archive filename/hash, commit/push/synchronize the completion documentation, and only then mark Phase 5A complete or begin Phase 5B.
5. Preserve the Phase 4B single-reference quality boundary: Phase 5A is a generation-correctness and performance phase, not a caption-quality claim.

## 12. 许可证与声明

本项目采用 Apache-2.0 License。VisionZip 论文、模型和官方仓库的权利归原作者及其许可声明所有。本仓库用于学习和可复现研究，不将尚未完成的路径描述为完整原生 Jittor 端到端复现。
