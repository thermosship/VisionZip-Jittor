# 第一阶段算法对齐说明

## 输入

| 张量 | CLIP ViT-L/14-336典型Shape | 含义 |
|---|---:|---|
| `hidden_states` | `[B,577,1024]` | CLS加576个Patch Token |
| `attentions` | `[B,16,577,577]` | 倒数第二层多头Attention |
| `metric` | `[B,577,64]` | 倒数第二层Mean-head Key |

## Dominant Token Selection

官方参数`dominant=54`表示54个图像Patch。实现另外保留CLS，所以Dominant部分实际有55个Token。

Top-k索引按注意力分数返回，但官方使用布尔Mask提取Hidden State。因此：

- `selected_indices`：CLS加按分数排序的Top-k索引；
- `dominant_ordered_indices`：真正输出时的原始序列顺序。

两者都被保存用于对齐。

## Contextual Token Merging

移除CLS和Dominant Patch后：

1. 对Key Metric做L2归一化；
2. 均匀选择Contextual Target；
3. 其余Token分配给余弦相似度最高的Target；
4. 使用One-hot矩阵完成批量聚合。

## 两种Merge语义

### `code_exact`（默认）

严格复现官方仓库：

```text
context = target + sum(assigned) / max(count, 1)
```

### `paper_avg`（消融）

将Target和分配Token一起求平均：

```text
context = (target + sum(assigned)) / (count + 1)
```

正式对齐默认只使用`code_exact`，`paper_avg`用于解释论文描述和代码行为之间的差异。

## Token预算口径

官方README中的`54 dominant + 10 contextual = 64`不包含CLS。当前实现的输出序列为：

```text
CLS + 54 dominant patches + 10 contextual = 65
```

因此README分别报告：

- nominal visual-token budget：64；
- actual CLIP output sequence：65。

128和192配置按官方54:10比例扩展，是本项目实验预设，不宣称为官方固定拆分。
