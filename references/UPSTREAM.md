# 上游参考版本

本项目的第一阶段PyTorch参考逻辑固定到以下官方仓库版本：

- Repository: `JIA-Lab-research/VisionZip`
- Branch: `main`
- Commit: `8f86b55c6f000eb033e6912538af2dd7dcb30502`
- Snapshot date: 2026-08-01
- License: Apache-2.0

核心对齐对象来自官方`visionzip/clip_encoder.py`：

1. 对CLS到图像Patch的多头注意力求和；
2. Top-k选择Dominant Patch；
3. 使用布尔Mask提取，因而输出保持原始Token顺序；
4. 使用倒数第二层Attention的Mean-head Key作为Metric；
5. 均匀选择Contextual Target；
6. 余弦相似度Argmax分配；
7. 官方代码执行`target_hidden + mean(assigned_hidden)`。

本项目没有复制官方LLaVA模型代码。`reference/pytorch_visionzip.py`是为数值对齐编写的独立、可测试参考模块。
