# 第二阶段：真实 CLIP 特征对齐

> 状态：已于 2026-08-02 在 AutoDL RTX 4090 上完成。64/128/192 三档均通过，离散索引与 Assignment 逐元素完全一致，流水线摘要为 `passed: true`。

## 1. 阶段目标

第一阶段使用固定随机张量验证 VisionZip 核心算法。第二阶段将输入替换为真实图片经过 CLIP ViT-L/14-336 得到的中间特征，并继续比较 PyTorch 与原生 Jittor：

- 倒数第二层 Hidden State；
- 倒数第二层 Attention；
- 同层 Key Metric；
- Dominant Token 索引；
- Contextual Target 与 Merge Assignment；
- Contextual Token 和最终压缩特征；
- Token 选择及语义合并可视化。

本阶段仍不加载 7B LLM，目的是先隔离并验证真实视觉编码器到 Token 压缩模块的边界。

## 2. 与官方代码的对应关系

固定参考上游：

```text
Repository: JIA-Lab-research/VisionZip
Commit: 8f86b55c6f000eb033e6912538af2dd7dcb30502
```

官方 CLIP 路径读取：

```text
attentions[-2]
hidden_states[-2]
encoder.layers[-2].metric
```

其中 `metric` 是倒数第二层 Self-Attention 中 `k_proj` 的输出，经以下变换得到：

```text
[B, N, hidden_dim]
→ [B, N, heads, head_dim]
→ [B, heads, N, head_dim]
→ mean(heads)
→ [B, N, head_dim]
```

对于 `openai/clip-vit-large-patch14-336`，典型 Shape 为：

```text
hidden_states: [B, 577, 1024]
attentions:    [B, 16, 577, 577]
metric:        [B, 577, 64]
patch grid:    24 × 24
```

`scripts/export_real_clip_reference.py` 使用 `k_proj` Forward Hook 捕获原始 Key Projection，不修改 CLIP 权重，也不依赖完整 LLaVA。

## 3. 两套 Python 环境的职责

### PyTorch 基准环境

```text
/root/miniconda3/bin/python
```

负责：

- 加载 CLIP 和图像预处理器；
- 提取真实 Hidden State、Attention 和 Key Metric；
- 运行 PyTorch VisionZip 参考实现；
- 保存 NPZ；
- 生成 Token 可视化。

### Jittor 环境

```text
/root/autodl-tmp/envs/visionzip-jittor/bin/python
```

负责：

- 读取完全相同的真实 CLIP NPZ；
- 使用原生 Jittor 执行 Token 选择和合并；
- 输出逐项数值对齐报告。

这保证 CLIP 特征只提取一次，PyTorch 与 Jittor 得到完全相同的输入。

## 4. AutoDL 准备

### 4.1 激活 Jittor 工作环境

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/envs/visionzip-jittor
export OMP_NUM_THREADS=8
cd /root/autodl-tmp/VisionZip-Jittor
```

### 4.2 安装 PyTorch 基准环境依赖

只给基准环境安装 Transformers 和 Pillow，不要将 Transformers 安装到 Jittor 环境：

```bash
/root/miniconda3/bin/python -m pip install -r requirements/real_clip.txt
```

验证现有 CUDA PyTorch 没有被替换：

```bash
/root/miniconda3/bin/python - <<'PY'
import torch
import transformers
from PIL import Image

print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("transformers:", transformers.__version__)
print("Pillow:", Image.__version__)
PY
```

预期 PyTorch 仍为 `2.1.2+cu118`，Transformers 为 `4.31.0`。

### 4.3 将 Hugging Face 缓存放到数据盘

```bash
mkdir -p /root/autodl-tmp/cache/huggingface
export HF_HOME=/root/autodl-tmp/cache/huggingface
export TRANSFORMERS_CACHE=/root/autodl-tmp/cache/huggingface/transformers
```

如果 Hugging Face 直连下载超时，可在当前终端临时设置可用的镜像端点；模型路径参数本身保持不变。

## 5. 创建可重复样例图

```bash
/root/miniconda3/bin/python scripts/create_sample_images.py \
  --output-dir assets/sample_images
```

生成：

```text
assets/sample_images/scene.png
assets/sample_images/text.png
assets/sample_images/dense.png
```

三张图分别强调自然场景、OCR 文本和密集局部元素。也可以用 `--image` 传入自己的图片。

## 6. 一键运行真实特征对齐

保持当前 Jittor 环境激活，执行：

```bash
python scripts/run_real_clip_pipeline.py \
  --torch-python /root/miniconda3/bin/python \
  --jittor-python /root/autodl-tmp/envs/visionzip-jittor/bin/python \
  --image-dir assets/sample_images \
  --model-name-or-path openai/clip-vit-large-patch14-336 \
  --cache-dir /root/autodl-tmp/cache/huggingface \
  --device cuda \
  --dtype float32
```

流水线只加载一次 CLIP，然后依次导出 64、128、192 三档参考结果，再分别调用 Jittor 对齐和可视化脚本。

## 7. 预期产物

### 真实特征与参考结果

```text
outputs/real_clip/manifest.json
outputs/real_clip/reference_clip_64_code_exact_float32_real_clip.npz
outputs/real_clip/reference_clip_128_code_exact_float32_real_clip.npz
outputs/real_clip/reference_clip_192_code_exact_float32_real_clip.npz
```

每个 NPZ 包含：

- `pixel_values`；
- `hidden_states`；
- `attentions`；
- `metric`；
- PyTorch 压缩输出和所有对齐中间量；
- 原图文件名、模型、层号、处理器参数、上游 Commit 和依赖版本。

### Jittor 对齐日志

```text
logs/real_clip/export_real_clip.log
logs/real_clip/alignment_clip_64_code_exact_float32.json
logs/real_clip/alignment_clip_128_code_exact_float32.json
logs/real_clip/alignment_clip_192_code_exact_float32.json
logs/real_clip/pipeline_summary.json
```

### 可视化

```text
outputs/real_clip/visualizations/*.png
```

每张图包含三栏：

1. CLIP 实际预处理输入；
2. CLS Attention 热度与 Dominant Token 边框；
3. Contextual Merge 分组、Contextual Target 和 Dominant Token。

## 8. 通过标准

FP32 正式实验使用：

```text
atol = 1e-5
rtol = 1e-5
```

以下离散结果必须逐元素完全一致：

- `selected_indices`；
- `dominant_ordered_indices`；
- `remaining_indices`；
- `target_positions`；
- `merge_positions`；
- `assignments`。

以下浮点结果必须通过 `allclose`：

- `compressed_tokens`；
- `contextual_tokens`；
- `cls_attention_sum`；
- `assignment_counts`。

任意档位失败时，流水线会以非零状态退出，不会写出成功摘要。

## 9. AutoDL 正式实验结果

### 9.1 实验环境

```text
Date: 2026-08-02
OS: Ubuntu 22.04.1 LTS
GPU: NVIDIA GeForce RTX 4090 24GB
CUDA Toolkit: 11.8.89
PyTorch: 2.1.2+cu118
Jittor: 1.3.11.0
Transformers: 4.31.0
Model: openai/clip-vit-large-patch14-336
Dtype: float32
Device: cuda
Tolerance: atol=1e-5, rtol=1e-5
```

测试批次由三张确定性样例图组成：

```text
assets/sample_images/dense.png
assets/sample_images/scene.png
assets/sample_images/text.png
```

真实输入 Shape：

```text
pixel_values:  [3, 3, 336, 336]
hidden_states: [3, 577, 1024]
attentions:    [3, 16, 577, 577]
metric:        [3, 577, 64]
patch grid:    24 × 24
```

CLIP 使用倒数第二个 Encoder Layer，即 `layer_index=22`、`hidden_states_index=23` 和 `attention_index=22`。Metric 来源为 `k_proj -> [B,H,N,D] -> mean(heads)`。

### 9.2 三档汇总

| 名义预算 | 实际输出 Shape | Compressed 最大误差 | Contextual 最大误差 | CLS Attention 最大误差 | Assignment Counts 最大误差 | Assignment | 结果 |
|---:|---:|---:|---:|---:|---:|---:|---|
| 64 | `[3, 65, 1024]` | `5.7220458984375e-06` | `5.7220458984375e-06` | `2.384185791015625e-07` | `0.0` | exact, 1.0 | PASS |
| 128 | `[3, 129, 1024]` | `1.9073486328125e-06` | `1.9073486328125e-06` | `2.384185791015625e-07` | `0.0` | exact, 1.0 | PASS |
| 192 | `[3, 193, 1024]` | `1.9073486328125e-06` | `1.9073486328125e-06` | `3.5762786865234375e-07` | `0.0` | exact, 1.0 | PASS |

三个预算下，下列离散结果全部逐元素完全一致，`agreement=1.0`：

- `selected_indices`；
- `dominant_ordered_indices`；
- `remaining_indices`；
- `target_positions`；
- `merge_positions`；
- `assignments`。

`compressed_tokens`、`contextual_tokens`、`cls_attention_sum` 和 `assignment_counts` 全部通过 `atol=1e-5, rtol=1e-5`。`logs/real_clip/pipeline_summary.json` 最终记录：

```json
{
  "passed": true
}
```

流水线同时生成 9 张可视化，即三张输入图片分别对应 64、128、192 三档结果。

## 10. Near-tie 数值问题与精确 CUDA 修复

### 10.1 初始现象

第一轮真实 CLIP 对齐中，Dominant Token 相关索引全部精确一致，但 64 档 Assignment 出现：

```text
assignment mismatch: 4 / 1536
assignment agreement: 0.9973958333333334
compressed_tokens max_abs_error: 0.012266159057617188
```

这些差异不是算法、索引、Batch Matrix Multiplication 或 Argmax 实现错误，而是部分真实 Key Metric 的余弦相似度非常接近。L2 归一化阶段极小的 float32 舍入差异改变了 near-tie 的最终大小关系，进而改变离散 Assignment。

### 10.2 数值路径隔离

逐阶段替换 PyTorch/Jittor 中间量得到：

```text
metric_filtered max error:                  0.0
Jittor norm max error:                      2.861022949219e-06
Jittor normalized max error:                8.940696716309e-08
Jittor norm + Jittor BMM mismatch:           4
PyTorch norm + Jittor BMM mismatch:          3
PyTorch normalized + Jittor BMM mismatch:    0
PyTorch operands + Jittor BMM mismatch:      0
PyTorch similarity + Jittor argmax mismatch: 0
```

因此可以确定：

- Gather 路径精确；
- Jittor BMM 在相同 operands 下精确；
- Jittor Argmax 在相同 similarity 下精确；
- 差异完全来自 L2 norm 归约和除法的数值路径。

### 10.3 修复方案

`visionzip_jittor/core.py` 增加原生 `jt.code` CUDA 路径 `_torch_cuda_l2_normalize_64`。该路径针对 CLIP ViT-L/14-336 的 `[B,N,64]` FP32 Metric：

1. 使用 32 个 CUDA lane，每个 lane 处理两个元素；
2. 按 PyTorch 2.1 CUDA 的 ascending warp shuffle 顺序归约；
3. 使用 `__fmul_rn` 固定平方舍入；
4. 使用 `__fadd_rn` 固定累加舍入；
5. 使用 `__fsqrt_rn` 固定平方根舍入；
6. 使用 `__fdiv_rn` 固定除法舍入。

最终精确诊断结果：

```text
norm max error:         0.0
norm exact elements:    1566 / 1566
normalized max error:   0.0
normalized exact:       100224 / 100224
similarity max error:   0.0
similarity exact:       15360 / 15360
assignment mismatch:    0
assignment agreement:   1.0
```

该修复对应提交：

```text
df23929 fix: match PyTorch CUDA normalization for CLIP metrics
```

精确路径仅在以下条件同时满足时启用：

```text
CUDA enabled
float32
3-D tensor
last dimension = 64
eps = 0.0
```

CPU、其他 dtype、其他 Metric 维度或 `eps > 0` 时继续使用通用原生 Jittor 归一化回退，不会错误套用特定 CUDA Kernel。

## 11. 结果归档

第二阶段证据包保存了流水线摘要、三档 JSON 对齐报告、诊断日志、Manifest、三张输入图片和九张可视化。外部归档文件的校验信息为：

```text
File: VisionZip-Jittor-phase2-evidence-20260802.tar.gz
SHA256: 8886E0FE914A0D68AEC70346005853DC83A9086185D58DCFE945D040DE612CDC
Entries: 35
Visualization PNGs: 9
```

大型真实 CLIP NPZ 不提交到 Git；仓库通过脚本、固定模型路径、固定样例图和 Manifest 保持实验可重复性。

## 12. 当前边界与下一阶段

本阶段可以证明：

- 官方 CLIP 类型的真实视觉特征能够正确进入 Jittor VisionZip；
- PyTorch 与 Jittor 的真实特征压缩行为一致；
- 三档 Token 选择、目标选择和 Merge Assignment 精确一致；
- Token 选择和语义合并可以可视化解释。

本阶段仍不能证明：

- 完整 LLaVA 回答质量；
- LLM Prefill 加速比例；
- 完整模型峰值显存下降；
- Projector 微调效果。

下一阶段接入多模态 Projector 与冻结 LLM。首先完成 64/128/192 三档最小 Forward/Backward，确认 VisionZip 输出确实进入 Projector、CLIP 与 LLM 保持冻结、Projector 获得梯度；最小集成通过后，再开展只训练 Projector 的高效微调和完整性能评估。
