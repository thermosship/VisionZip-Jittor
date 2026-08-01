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

## 结论

在固定随机输入和 FP32 精度下，原生 Jittor 实现复现了官方 PyTorch CLIP 路径的 Dominant Token Selection 与 Contextual Token Merging。三个 Token 预算的离散决策完全一致，最终特征差异处于单精度浮点舍入量级。

本结果只证明第一阶段核心算法的框架对齐，不等同于真实图像、完整视觉编码器或 LLaVA 端到端结果；真实特征和端到端验证属于后续阶段。
