# 提交准备审计清单

> 状态日期：2026-08-03。该清单面向培育期 PPT/视频唯一一次提交，PPT 和视频制作暂未开始。

## 1. 选题合规

- [x] 论文：**VisionZip: Longer is Better but Not Necessary in Vision Language Models**；
- [x] 会议：CVPR 2025，满足“发表时间不超过 2 年”；
- [x] 使用原生 Jittor 实现核心算法、Projector 训练与 GPT-2 路径；
- [x] 2026-08-03 对 `GrokCV/Jittor-Sprouts` 当前 HEAD `451dd0e5499d2a5730d9314e7bba0b8320d6afc2` 做仓库内全文检索，未发现 `VisionZip`；
- [ ] **提交当天必须再次检查 Jittor-Sprouts 列表**，因为列表可能更新；
- [ ] 提交前确认 GitHub 仓库保持 public，release/tag 可访问。

## 2. README 与代码入口

- [x] 环境安装；
- [x] 数据准备脚本；
- [x] PyTorch 参考导出；
- [x] Jittor 对齐脚本；
- [x] 训练脚本和 resume；
- [x] 测试命令；
- [x] 性能命令和协议；
- [x] 结果表、Loss 曲线、Token 可视化；
- [x] 训练范围和 non-claims；
- [x] 机器可读 compact logs；
- [ ] 在一台干净 AutoDL 实例执行一次 README smoke walkthrough，并保存最终日志；
- [ ] 创建 submission tag/release 后再次检查所有相对链接。

## 3. 已解决的主要技术风险

| 风险 | 处理结果 |
|---|---|
| CLIP near-tie 导致 Assignment 不一致 | 实现 PyTorch-compatible CUDA FP32 norm，三档 Assignment 100% exact |
| PyTorch/Jittor 框架环境变量冲突 | 分离 export 与 Jittor 环境，记录 `USE_TORCH` 风险 |
| Projector-only 训练误更新 GPT-2 | stop-grad、optimizer scope、训练前后 SHA256 三重检查 |
| Jittor `projector.eval()` 改变 stop-grad | evaluation 后显式 `projector.train()` 并加回归测试 |
| checkpoint 只能保存权重不能恢复 optimizer | 保存完整 Projector/Adam 状态并验证 resume |
| 只报告 train loss、没有 held-out 证据 | 使用固定 1,024 样本、10,744 target tokens 的 13 次全量评估 |
| 性能统计混入启动/评估/I/O | 冻结 post-warm-up optimizer-step protocol，并单独定义显存采样范围 |
| KV-cache raw logits 浮点漂移 | 使用 exact greedy/cache contract + frozen TV gate，raw logits 保留为 diagnostic |
| 原始结果只存在终端/AutoDL | 跨主机证据包、内部 SHA256、Git 中 compact CSV/JSON/PNG 三层保存 |
| README 缺少提交视角 | 重构 README，增加结果总览、训练边界和提交清单 |

## 4. 仍需在提交前完成的材料风险

1. **干净环境 walkthrough**：按 README 从 install/test 到小规模 alignment/training smoke 完整走一遍；大模型/数据下载可引用已校验证据包，但命令必须无误。
2. **GitHub Release**：普通源码仓库不上传权重和数据；可发布小型 evidence archive，或在 release notes 中提供哈希和获取说明。
3. **PPT/视频**：尚未开始；应直接复用 `docs/assets` 图表，不要重新手抄数值。
4. **提交日列表复核**：再次检查 Jittor-Sprouts 是否新增 VisionZip。
5. **最终声明审计**：全文搜索 `full reproduction`、`SOTA`、`LLaVA-equivalent`、`bitwise exact`、`universal speedup` 等越界表述。

## 5. 证据包

| 阶段 | 文件 | SHA256 |
|---|---|---|
| Phase 2 | `VisionZip-Jittor-phase2-evidence-20260802.tar.gz` | `8886e0fe914a0d68aec70346005853dc83a9086185d58dcfe945d040de612cdc` |
| Phase 4A | `VisionZip-Jittor-phase4a-evidence-20260803.tar.gz` | `01942f1fd7e82faf6eb5e8bcb9ffca9c2474b50718eea47e00ba446960926858` |
| Phase 4B final-v2 | `VisionZip-Jittor-phase4b-evidence-final-v2-20260803.tar.gz` | `2efeeaa88f18ab11b8431a7dd810b296366073d14b5717d02c72152dba70c032` |
| Phase 5A | `VisionZip-Jittor-phase5a-evidence-20260803.tar.gz` | `20093fb7550d6e17fc96566191236bc3631952998a5d4097d245ae1f2037ec81` |

上述哈希也记录在 [`submission_results.json`](results/submission_results.json)。证据包不是运行依赖，复现者可按 README 重新生成对应 artifacts。

## 6. 当前完成度

- Jittor 技术复现：**约 96%–98%**；
- 培育期代码与实验材料：**约 93%–96%**；
- 完整提交（含 PPT、讲稿、视频）：**约 80%–85%**。

剩余工作以“干净环境 walkthrough + PPT/视频”为主，不建议在唯一提交前临时扩展到完整 LLaVA 或论文全部 benchmark。
