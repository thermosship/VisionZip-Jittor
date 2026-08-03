# 干净环境 README Walkthrough

> 最终状态：**PASS**<br>
> 执行日期：2026-08-03（Asia/Shanghai）<br>
> 固定提交：`04f098d8d69edd0ad85da37351892a3685ac475b`<br>
> 机器可读摘要：[`results/clean_readme_walkthrough_04f098d.json`](results/clean_readme_walkthrough_04f098d.json)

## 1. 目的与验收边界

本 walkthrough 用于在冻结 submission tag/release 前，从 GitHub 创建独立 checkout，并从空 conda prefix 按 README 安装 Jittor 依赖，验证公开仓库中的环境、测试、PyTorch/Jittor 对齐、真实 CLIP、真实 GPT-2 smoke 和 Phase 4B 数据预检命令能够执行。

它不是对所有昂贵实验的重复运行。本次没有重新下载/物化 8,192 样本数据集，没有重跑 1,344-step Phase 4B 正式训练，也没有重跑 Phase 5A 正式 KV-cache benchmark；这些结论由已固定的 compact logs、文档和外部 SHA256 证据包覆盖。

## 2. 独立 checkout 与 fresh environment

| 项目 | 值 |
|---|---|
| Checkout | `/root/autodl-tmp/submission_walkthrough/VisionZip-Jittor-04f098d` |
| Git 状态 | detached HEAD at `04f098d8d69edd0ad85da37351892a3685ac475b` |
| Fresh conda prefix | `/root/autodl-tmp/envs/visionzip-readme-clean-04f098d` |
| Shared cache | `/root/autodl-tmp/cache`；只复用 Hugging Face/PIP 下载缓存，不复用生成结果 |
| Python | `3.10.20` |
| Jittor | `1.3.11.0` |
| PyTorch reference | `2.1.2+cu118` |
| Transformers | `4.31.0` |
| CUDA toolkit used by Jittor | `11.8.89` |
| Driver | `580.105.08` |
| GPU | NVIDIA GeForce RTX 4090, 24564 MiB |
| g++ | `11.3.0` |

Fresh prefix 的安装命令与 README 一致：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda create -p /root/autodl-tmp/envs/visionzip-readme-clean-04f098d \
  python=3.10 -y
conda activate /root/autodl-tmp/envs/visionzip-readme-clean-04f098d
python -m pip install -r requirements/jittor.txt
python -m pip install -r requirements/phase3b_jittor.txt
python -m pip install -r requirements/dev.txt
python -m pip install -e .
```

随后验证 checkout-relative 激活合同：

```bash
export VISIONZIP_JITTOR_ENV=/root/autodl-tmp/envs/visionzip-readme-clean-04f098d
export VISIONZIP_CACHE_ROOT=/root/autodl-tmp/cache
source environment/activate_jittor.sh
```

实际 `pwd` 保持在独立 checkout，`python` 指向 fresh prefix；`bash -n environment/activate_jittor.sh` 通过。

## 3. 执行项目

### 3.1 静态入口和测试

执行了 README 涉及的 11 个关键脚本 `--help` smoke，包括 alignment、real CLIP、GPT-2 export/smoke、Phase 4B prepare/features/training、Phase 5A 和 submission asset builder。

```bash
python -m unittest discover -s tests -v
python -m compileall -q scripts visionzip_jittor reference tests
git diff --check
```

结果：`Ran 80 tests in 23.385s`，`OK (skipped=8)`；8 个 skip 是 fresh Jittor prefix 中未安装 PyTorch 的独立 PyTorch-reference 测试，PyTorch 参考命令按 README 使用 `/root/miniconda3/bin/python` 单独执行。`compileall` 和 diff check 均通过。

### 3.2 合成 PyTorch/Jittor 对齐

```bash
/root/miniconda3/bin/python scripts/export_pytorch_reference.py \
  --config configs/visionzip_64.json \
  --output outputs/readme_walkthrough/reference_64.npz

python scripts/run_jittor_alignment.py \
  --reference outputs/readme_walkthrough/reference_64.npz \
  --output-json "$LOG_DIR/synthetic/alignment_64.json"
```

结果：`passed=true`，Assignment `exact=true`、agreement `1.0`；compressed token max abs error `2.384185791015625e-07`，allclose。

### 3.3 确定性样例图与真实 CLIP

Fresh clone 不包含二进制样例图，因此先执行已补入 README 的确定性生成命令：

```bash
/root/miniconda3/bin/python scripts/create_sample_images.py \
  --output-dir assets/sample_images
```

随后从共享的只读 Hugging Face 模型缓存运行 64/128/192 三档真实 CLIP pipeline。三档均 `passed=true`，Assignment 全部 exact：

| Budget | Assignment agreement | Compressed max abs error | allclose |
|---:|---:|---:|---|
| 64 | `1.0` | `5.7220458984375e-06` | true |
| 128 | `1.0` | `1.9073486328125e-06` | true |
| 192 | `1.0` | `1.9073486328125e-06` | true |

### 3.4 真实 GPT-2 export 与原生 Jittor smoke

从共享模型下载缓存重新导出 GPT-2 Small FP32 artifacts 到 clean checkout 的 untracked 输出目录，再运行原生 Jittor smoke：

- architecture：`GPT2LMHeadModel`；
- parameters：`124,439,808`；
- reference logit max abs error：`0.00020599365234375`，allclose at `atol=rtol=5e-4`；
- 64/128/192：全部 `passed=true`；
- 三档均 Assignment exact、Projector gradient finite、Projector 参数发生更新；
- frozen language model 与 optimizer-scope gate 通过。

### 3.5 Phase 4B 数据 preflight

```bash
/root/miniconda3/bin/python scripts/prepare_phase4b_dataset.py \
  --config configs/phase4b_commoncatalog_cc_by_8k.json
```

结果：`disk_preflight_passed=true`，目标 `8,192` 样本、`7,168 / 1,024` split，预估所需 `9,003,935,588` bytes，当时可用 `203,333,971,968` bytes。命令没有下载或物化数据集。

## 4. Walkthrough 发现并修复的问题

Walkthrough 不是只做“成功演示”，而是实际发现了两个公开复现风险：

1. **激活脚本跳回主仓库**：原脚本硬编码 checkout 和 conda prefix。`a02b9cf` 改为从 `BASH_SOURCE[0]` 推导项目根目录，并支持 `VISIONZIP_JITTOR_ENV` / `VISIONZIP_CACHE_ROOT`。
2. **README 缺少样例图生成步骤**：PNG 按设计不提交到 Git，fresh clone 的 real-CLIP 命令因此找不到输入。`04f098d` 在 README 中加入 `scripts/create_sample_images.py`，并增加测试保证生成命令位于 real-CLIP pipeline 之前。

首次辅助 orchestration 还把不存在的 `scripts/export_phase4b_clip_features.py` 写入 help 列表；修正为仓库实际入口 `scripts/precompute_phase4b_features.py` 后继续。这是 walkthrough 辅助脚本错误，不是仓库或 README 缺陷。失败尝试日志仍保存在 AutoDL，未覆盖。

## 5. 输出、Git 状态与证据

最终 clean checkout 的 tracked tree 未被修改。预期 untracked 项只有：

```text
outputs/readme_walkthrough/
visionzip_jittor.egg-info/
```

前者是本次生成的 NPZ/GPT-2 artifacts，后者由 `pip install -e .` 产生；两者均不提交。

完整控制台和逐步骤日志保存在 AutoDL：

```text
/root/autodl-tmp/VisionZip-Jittor/logs/submission_readiness/clean_walkthrough_04f098d/
```

该目录保持 untracked。机器可读 summary 的外部 SHA256：

```text
5b8ebf154a72d6c3951910330fcdadc250611fcaa1461c30fb5b70fa6cc03f01
```

提交到 Git 的 compact JSON 记录了每个外部日志文件的 SHA256，但不包含模型权重、CLIP reference NPZ、数据集、feature shards、checkpoint 或大控制台日志。

## 6. 最终结论

`clean_readme_walkthrough_v1.passed = true`。

因此，README 的环境激活、fresh-prefix 安装、测试、合成对齐、真实 CLIP、真实 GPT-2 smoke 和 Phase 4B preflight 已在独立 GitHub checkout 上通过。完成本文档审计并提交后，可以冻结 immutable submission tag/release；release 不附加模型、数据、checkpoint 或大日志。
