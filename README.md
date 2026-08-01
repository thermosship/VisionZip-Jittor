# VisionZip-Jittor

使用 **Jittor 原生张量算子**复现 VisionZip 的视觉 Token 压缩核心，并建立可重复的 PyTorch/Jittor 数值对齐流程。

> 当前进度：第一阶段核心算法已完成 FP32 数值对齐与 RTX 4090 核心性能测试；下一步进行真实 CLIP 视觉特征验证。

## 1. 项目目标

本项目计划使用 Jittor 复现 VisionZip，最终覆盖：

- Dominant Token Selection；
- Contextual Token Merging；
- 与官方 PyTorch 实现的算子级对齐；
- 训练无关的多模态推理；
- 仅训练多模态 Projector 的高效微调；
- 训练、测试、性能和可视化日志。

当前仓库处于第一阶段，已经实现：

- 原生 Jittor VisionZip 核心；
- 独立 PyTorch 参考实现；
- 64、128、192 Token 配置；
- PyTorch 参考张量导出；
- Jittor 中间变量逐项对齐；
- 单元测试和核心性能测试入口。

## 2. 上游版本与复现范围

第一阶段参考逻辑固定到：

```text
Repository: JIA-Lab-research/VisionZip
Branch: main
Commit: 8f86b55c6f000eb033e6912538af2dd7dcb30502
Snapshot date: 2026-08-01
```

详细版本记录见 [`references/UPSTREAM.md`](references/UPSTREAM.md)，第一阶段数值结果见 [`docs/PHASE1_RESULTS.md`](docs/PHASE1_RESULTS.md)。

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

## 9. 项目结构

```text
VisionZip-Jittor/
├── configs/                    # 64/128/192 Token 配置
├── docs/                       # 算法和实验结果说明
├── environment/                # AutoDL 激活和环境证据脚本
├── logs/                       # 精简实验日志
├── outputs/                    # 中间张量，默认不提交
├── reference/                  # PyTorch 参考实现
├── references/                 # 上游版本记录
├── requirements/               # 两个环境的依赖说明
├── scripts/                    # 导出、对齐和性能测试脚本
├── tests/                      # 单元测试
└── visionzip_jittor/
    ├── config.py
    └── core.py                 # 原生 Jittor 核心
```

## 10. 第一阶段状态

- [x] PyTorch 参考实现与固定的官方 CLIP 代码逻辑一致；
- [x] 原生 Jittor 核心实现完成；
- [x] PyTorch 侧单元测试通过；
- [x] AutoDL 上 Jittor 小张量执行测试通过；
- [x] FP32 中间索引 100% 一致；
- [x] FP32 Assignment 100% 一致；
- [x] 浮点结果在指定容差内通过；
- [x] 64/128/192 三种配置完成对齐；
- [x] 完成 RTX 4090 核心性能测试并保存摘要；
- [ ] 使用真实 CLIP 特征再次验证。

## 11. 后续阶段

1. 从官方 CLIP 导出真实图像的 Hidden State、Attention 和 Key Metric；
2. 使用 Jittor 对真实视觉特征进行对齐；
3. 接入多模态 Projector 和冻结 LLM；
4. 实现只训练 Projector 的高效微调；
5. 对齐 Loss、效果、速度、显存和可视化；
6. 整理最终 README、PPT 和视频素材。

## 12. 许可证与声明

本项目采用 Apache-2.0 License。VisionZip 论文、模型和官方仓库的权利归原作者及其许可声明所有。本仓库用于学习和可复现研究，不将尚未完成的路径描述为完整原生 Jittor 端到端复现。
