# 第一阶段数值对齐结果

## 实验环境

- Date: 2026-08-01
- GPU: NVIDIA GeForce RTX 4090 24GB
- Driver: 570.124.04
- CUDA Toolkit: 11.8.89
- PyTorch: 2.1.2+cu118
- Jittor: 1.3.11.0
- Precision: FP32
- Random seed: 2026

## 输入 Shape

使用 CLIP ViT-L/14-336 路径的典型张量 Shape：

```text
hidden_states: [1, 577, 1024]
attentions:    [1, 16, 577, 577]
metric:        [1, 577, 64]
```

## 对齐结果

| 名义预算 | 实际输出（含 CLS） | 压缩 Token 最大绝对误差 | Contextual Token 最大绝对误差 | 索引一致率 | Assignment 一致率 | 结果 |
|---:|---:|---:|---:|---:|---:|---|
| 64 | 65 | 2.3841858e-07 | 2.3841858e-07 | 100% | 100% | PASS |
| 128 | 129 | 2.3841858e-07 | 2.3841858e-07 | 100% | 100% | PASS |
| 192 | 193 | 2.3841858e-07 | 2.3841858e-07 | 100% | 100% | PASS |

以下索引均与 PyTorch 参考结果逐元素完全相等：

- `selected_indices`；
- `dominant_ordered_indices`；
- `remaining_indices`；
- `target_positions`；
- `merge_positions`；
- `assignments`。

`assignment_counts` 的最大和平均绝对误差均为 0。所有浮点张量均在 `atol=1e-5, rtol=1e-5` 下通过 `allclose`。

## Jittor 核心性能测试

### 测试配置

- GPU: NVIDIA GeForce RTX 4090 24GB；
- Batch Size: 1；
- Precision: FP32；
- 输入 Shape：`hidden_states=[1,577,1024]`、`attentions=[1,16,577,577]`、`metric=[1,577,64]`；
- Warmup: 20 次；
- Timed iterations: 100 次；
- 测量范围：Jittor VisionZip 核心 Token 选择与合并，不包括 CLIP 编码、Projector、LLM Prefill 和文本生成。

### 测试结果

| 名义预算 | 实际输出（含 CLS） | 相对 577 Token 的序列缩减率 | Mean (ms) | Median (ms) | Min (ms) | Max (ms) | Std (ms) | 约合吞吐量 (calls/s) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 65 | 88.73% | 0.977824 | 0.919659 | 0.868969 | 1.734540 | 0.172511 | 1022.68 |
| 128 | 129 | 77.64% | 0.924812 | 0.870302 | 0.806093 | 1.998819 | 0.194281 | 1081.30 |
| 192 | 193 | 66.55% | 0.943357 | 0.892080 | 0.852719 | 1.586370 | 0.153472 | 1059.00 |

吞吐量按 `1000 / mean_latency_ms` 计算。序列缩减率按 `1 - actual_output_tokens / 577` 计算，实际输出包含 CLS Token。

### 结果解读

1. 三种预算的核心压缩延迟均约为 `0.9–1.0 ms`，说明 Jittor 原生实现可在 RTX 4090 上以毫秒级完成该模块。
2. 三档延迟没有呈现严格单调关系。64 Token 配置需要合并更多剩余 Token，而更高预算会增加 Dominant Token 的收集与输出量，两部分开销可能相互抵消。
3. 128 Token 的本次均值低于 64 Token，但差值约为 `0.053 ms`，小于各组标准差且可能受 GPU 调度、同步和测量抖动影响，不能据此认定 128 Token 配置在一般情况下更快。
4. 该结果只衡量压缩核心开销。VisionZip 的主要收益发生在后续 LLM Prefill 中，因此必须在接入真实 CLIP 与 LLM 后再报告端到端延迟、峰值显存和生成吞吐量。

## 结论

在固定随机输入和 FP32 精度下，原生 Jittor 实现复现了官方 PyTorch CLIP 路径的 Dominant Token Selection 与 Contextual Token Merging。三个 Token 预算的离散决策完全一致，最终特征差异处于单精度浮点舍入量级；RTX 4090 上的核心模块平均执行时间均低于 1 ms。

本结果只证明第一阶段核心算法的框架对齐，不等同于真实图像、完整视觉编码器或 LLaVA 端到端结果；真实特征和端到端验证属于后续阶段。
