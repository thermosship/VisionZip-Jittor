# VisionZip-Jittor

使用 **Jittor 原生张量算子**复现 VisionZip 的视觉 Token 压缩核心，并建立可重复的 PyTorch/Jittor 数值对齐流程。

> Current status: Phases 1, 2, and Phase 3A are complete. Phase 3B native Jittor GPT-2 code, artifact export, real generation, Projector-only backward, and measurement runner are implemented; formal AutoDL CUDA validation is pending.

## 1. 项目目标

本项目计划使用 Jittor 复现 VisionZip，最终覆盖：

- Dominant Token Selection；
- Contextual Token Merging；
- 与官方 PyTorch 实现的算子级对齐；
- 训练无关的多模态推理；
- 仅训练多模态 Projector 的高效微调；
- 训练、测试、性能和可视化日志。

当前仓库已完成第一阶段与第二阶段，已经实现：

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
- 真实 CLIP 特征在 64/128/192 三档下的逐项对齐与 9 张 Token 可视化。

## 2. 上游版本与复现范围

第一阶段参考逻辑固定到：

```text
Repository: JIA-Lab-research/VisionZip
Branch: main
Commit: 8f86b55c6f000eb033e6912538af2dd7dcb30502
Snapshot date: 2026-08-01
```

详细版本记录见 [`references/UPSTREAM.md`](references/UPSTREAM.md)，第一阶段数值结果见 [`docs/PHASE1_RESULTS.md`](docs/PHASE1_RESULTS.md)，第二阶段真实 CLIP 实测结果与运行说明见 [`docs/PHASE2_REAL_CLIP.md`](docs/PHASE2_REAL_CLIP.md)。

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

Phase 3B now has an initial real-LLM implementation based on GPT-2 small. Unlike the Phase 3A stub, this path executes real token embeddings, learned positional embeddings, 12 causal Transformer blocks, final LayerNorm, and the tied GPT-2 LM head in native Jittor. The model hidden size is 768, so the Phase 3B Projector maps `1024 -> 768 -> 768`.

The implementation includes a Hugging Face float32 weight/tokenizer exporter, deterministic text-logit reference, checksums for weights/config/tokenizer/reference, native tensor-count and parameter-count integrity checks, a native NPZ weight loader, real causal language loss, Projector-only optimizer step, full frozen-language SHA256 verification, greedy decoded text, warmed repeated prefill timing, and sampled process GPU memory. See [`docs/PHASE3B_REAL_GPT2.md`](docs/PHASE3B_REAL_GPT2.md).

The code is implemented but is **not yet recorded as Phase 3B complete**. Completion requires running the AutoDL CUDA procedure and saving a top-level `real_llm: true`, `passed: true` report for 64/128/192. Generated text before Projector training is execution evidence only, not a visual-language quality result.

## 9. 项目结构

```text
VisionZip-Jittor/
├── assets/sample_images/       # 可重复样例图生成位置
├── configs/                    # 64/128/192 Token 配置
├── docs/                       # 算法和实验结果说明
├── environment/                # AutoDL 激活和环境证据脚本
├── logs/                       # 精简实验日志
├── outputs/                    # 中间张量，默认不提交
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
    `-- gpt2.py                 # native Jittor real GPT-2
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
- [ ] Run the Phase 3B real GPT-2 path on AutoDL for 64/128/192 and save a `real_llm: true`, `passed: true` report.

## 11. Next steps

1. Export the GPT-2 float32 NPZ, tokenizer, config, manifest, and deterministic PyTorch logit reference on AutoDL.
2. Run native Jittor GPT-2 unit tests and fix any environment-specific Jittor operator/API differences.
3. Validate budget 64 first, including text-logit alignment, real generation, freezing, and Projector gradients.
4. Run the full 64/128/192 Phase 3B report and archive JSON/log evidence with SHA256.
5. Only after the minimum real-LLM integration passes, implement Projector training, KV cache, mixed precision, and downstream evaluation.

## 12. 许可证与声明

本项目采用 Apache-2.0 License。VisionZip 论文、模型和官方仓库的权利归原作者及其许可声明所有。本仓库用于学习和可复现研究，不将尚未完成的路径描述为完整原生 Jittor 端到端复现。
