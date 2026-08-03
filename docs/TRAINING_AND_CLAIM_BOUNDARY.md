# 训练、评估与结论边界

本文件固定提交材料中的训练口径，避免把“工程路径跑通”“held-out loss 下降”和“论文全部质量结论复现”混为一谈。

## 1. 三类实验的角色

| 实验 | 数据 | 目的 | 是否用于质量结论 |
|---|---|---|---|
| Phase 4A tiny overfit | 3 张项目生成图，2 train / 1 validation | 检查 Projector-only optimizer、checkpoint 和 resume | 否，仅基础设施 smoke |
| Phase 4B real paired pilot | CommonCatalog CC-BY，8,192 样本 | 验证真实配对数据上的可训练性、held-out NLL 与性能 | 只报告固定 pilot 的 NLL/困惑度和单参考指标 |
| Phase 5A KV-cache | 3 张 Phase 2 测试图 | 验证 cached/uncached 生成决策、cache contract 和性能 | 否，不用于 caption quality |

Phase 4A 的训练 loss 从 `9.8257` 降到 `4.4797`，但单样本 validation loss 从 `7.5415` 到 `7.7522`，因此它只能证明 tiny-overfit 与训练基础设施。正式训练趋势使用 Phase 4B 的 1,024 样本 held-out 评估。

## 2. Phase 4B 数据边界

- 数据集：`common-canvas/commoncatalog-cc-by`；
- 固定 revision：`80f50fe4a1ca937f37a11be3f8eee5199d776ff3`；
- 目标样本数：8,192；
- 确定性划分：7,168 train / 1,024 validation；
- 每条样本保留 source、license 和 attribution 信息；
- caption 为数据集提供的单条 BLIP-2 synthetic caption；
- 不把这些 caption 描述为人工标注，也不把单参考 BLEU/ROUGE 与 COCO 多参考指标直接比较。

详细许可和过滤策略见 [`PHASE4B_DATASET_PLAN.md`](PHASE4B_DATASET_PLAN.md)。

## 3. 冻结和训练范围

训练图：

```text
precomputed CLIP/VisionZip tokens (frozen)
        -> Jittor Projector (trainable)
        -> native Jittor GPT-2 Small (frozen)
        -> target-only causal language loss
```

严格约束：

1. CLIP 不进入 Phase 4B backward；使用已校验的冻结视觉特征 shard；
2. GPT-2 124,439,808 个语言参数全部 stop-grad；
3. optimizer 参数集合必须严格等于 Projector 参数集合；
4. 每个 update 检查 loss/gradient finite；
5. 训练前后 GPT-2 参数 SHA256 必须一致；
6. checkpoint 包含完整 Projector 与 Adam 状态，并显式验证 resume；
7. evaluation 后必须恢复 `projector.train()`，防止 Jittor `eval()` 改变参数 stop-grad 状态。

## 4. 固定训练配置

| 参数 | 值 |
|---|---:|
| Vision token budget | 64（CLS 额外保留，实际 65） |
| Projector | `1024 -> 768 -> 768`, GELU |
| GPT-2 | `openai-community/gpt2`, 124M, FP32 |
| Micro batch | 4 |
| Gradient accumulation | 4 |
| Effective batch | 16 |
| Optimizer steps | 1,344 |
| Warm-up steps | 67 |
| Base learning rate | `1e-4` |
| Evaluation interval | 112 steps |
| Checkpoint interval | 112 steps |
| Seed | 2026 |

配置文件：[`configs/phase4b_commoncatalog_cc_by_8k.json`](../configs/phase4b_commoncatalog_cc_by_8k.json)。

## 5. 可复制命令

以下命令在 **AutoDL/Jupyter Terminal** 执行。长任务应放入 `tmux`。

### 5.1 环境

```bash
cd /root/autodl-tmp/VisionZip-Jittor
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/envs/visionzip-jittor
export OMP_NUM_THREADS=8
```

PyTorch/Hugging Face 导出必须使用 `/root/miniconda3/bin/python`，Jittor 训练必须使用 `/root/autodl-tmp/envs/visionzip-jittor/bin/python`，不要混用 framework environment flags。

### 5.2 数据准备

仅做空间/文件预检：

```bash
python scripts/prepare_phase4b_dataset.py \
  --config configs/phase4b_commoncatalog_cc_by_8k.json
```

实际下载和物化：

```bash
python scripts/prepare_phase4b_dataset.py \
  --config configs/phase4b_commoncatalog_cc_by_8k.json \
  --cache-dir /root/autodl-tmp/cache/huggingface \
  --execute
```

### 5.3 冻结特征

```bash
/root/miniconda3/bin/python scripts/precompute_phase4b_features.py \
  --config configs/phase4b_commoncatalog_cc_by_8k.json \
  --dataset-manifest datasets/phase4b/commoncatalog_cc_by_8k/manifest.json \
  --output-dir outputs/phase4b/commoncatalog_cc_by_8k/features \
  --cache-dir /root/autodl-tmp/cache/huggingface \
  --device cuda
```

已有 32 个 feature shards 时，先做 hash/identity 检查：

```bash
/root/miniconda3/bin/python scripts/precompute_phase4b_features.py \
  --config configs/phase4b_commoncatalog_cc_by_8k.json \
  --dataset-manifest datasets/phase4b/commoncatalog_cc_by_8k/manifest.json \
  --output-dir outputs/phase4b/commoncatalog_cc_by_8k/features \
  --verify-existing
```

### 5.4 训练

```bash
tmux new -s visionzip-phase4b

python scripts/run_phase4b_training.py \
  --config configs/phase4b_commoncatalog_cc_by_8k.json \
  --prepared-manifest datasets/phase4b/commoncatalog_cc_by_8k/manifest.json \
  --feature-manifest outputs/phase4b/commoncatalog_cc_by_8k/features/manifest.json \
  --artifact-dir outputs/phase3b/gpt2 \
  --output-dir outputs/phase4b/commoncatalog_cc_by_8k/training \
  --log-dir logs/phase4b/training
```

恢复训练：

```bash
python scripts/run_phase4b_training.py \
  --config configs/phase4b_commoncatalog_cc_by_8k.json \
  --prepared-manifest datasets/phase4b/commoncatalog_cc_by_8k/manifest.json \
  --feature-manifest outputs/phase4b/commoncatalog_cc_by_8k/features/manifest.json \
  --artifact-dir outputs/phase3b/gpt2 \
  --output-dir outputs/phase4b/commoncatalog_cc_by_8k/training_resume \
  --log-dir logs/phase4b/training_resume \
  --resume outputs/phase4b/commoncatalog_cc_by_8k/training_benchmark_0f53a93/checkpoints/projector_step_001120.npz
```

## 6. 训练日志与曲线

普通 Git 历史提交下列小型材料：

- [`phase4b_training_trace.csv`](results/phase4b_training_trace.csv)：1,344 步完整 compact training trace；
- [`phase4b_validation_curve.csv`](results/phase4b_validation_curve.csv)：13 次全量 held-out evaluation；
- [`phase4b_loss_curve.png`](assets/phase4b_loss_curve.png)：原始 train NLL、32-step 平滑和 held-out NLL；
- [`submission_results.json`](results/submission_results.json)：跨阶段机器可读摘要与源证据包 SHA256。

原始 `console.log`、JSONL、checkpoint、GPT-2 权重、CLIP NPZ、数据集和 feature shards 体积较大，保存在本地/AutoDL 和带内部 `SHA256SUMS` 的证据包，不进入 GitHub 主分支。

## 7. 指标解释

### 可以说

- “在固定 8,192 样本 licensed pilot 上，held-out target NLL 从 6.6437 降至 2.4132。”
- “训练仅更新 Projector，GPT-2 参数哈希保持不变。”
- “真实 CLIP 特征上的 VisionZip 三档离散选择和 Assignment 与 PyTorch 精确一致。”
- “在固定 RTX 4090/Jittor 协议下记录了训练吞吐、显存与 KV-cache latency。”

### 不可以说

- “已完整复现 VisionZip 论文全部实验结论。”
- “已达到或超过 LLaVA/VisionZip 原论文的 caption/VQA benchmark。”
- “生成质量已经达到人工 caption 水平。”
- “所有模型、硬件和长度上都能获得相同 KV-cache 加速。”
- “cached/uncached raw logits bitwise identical。”

## 8. 为什么这已经满足当前技术展示目标

当前仓库提供了环境、数据准备、训练、测试、PyTorch 对齐、性能、Loss 曲线、结果可视化和失败诊断链路。它是一个有明确边界的小规模 Jittor 复现，而不是把有限资源实验包装成论文全表复现。提交时应主动展示 near-tie 数值问题、Jittor `eval()` stop-grad 问题以及对应修复，这些内容能体现独立分析和实现能力。
