# Paper Outline

## 1. Title And Abstract

内容规划：

- 明确研究问题：attention weight 能否作为 token importance 的可靠代理。
- 说明实验对象：DistilBERT + SST-2。
- 简述方法：attention、`gradient × input`、`LOO` 三种排序；排序相似性分析；token 删除实验。
- 简述核心结果：attention 与替代方法相关性较弱，`LOO` 删除效果最强，attention 强于随机但弱于 `LOO`。

建议图：

- 无。摘要直接用文字给出问题、方法和主要发现即可。

## 2. Introduction

内容规划：

- 介绍 Transformer attention 常被直接当作解释信号的背景。
- 提出研究动机：高 attention 是否真的对应高因果影响或高预测敏感度。
- 给出本文的核心研究问题与假设。
- 明确本文贡献：
  1. 在一个小而完整的设定下复核 attention 的解释性；
  2. 使用三种 token 排序方式做对比；
  3. 用多 seed 和删除实验验证结论是否稳定。

建议图：

- 不强制放图。
- 如果第一页版面允许，可以把 `Fig. 1` 放在 Introduction 末尾或 Methods 开头，作为读者进入实验设计的总览图。

## 3. Related Work

内容规划：

- Attention as explanation。
- Gradient-based attribution in NLP。
- Leave-one-out / perturbation-based importance。
- 与现有争议的关系：本文不是提出新模型，而是在课程作业规模内做一个聚焦的实证检验。

建议图：

- 无。

## 4. Methods / Experimental Setup

内容规划：

- 数据集与任务：GLUE SST-2 二分类。
- 基础模型：`distilbert-base-uncased`。
- 训练设置：4 个 seed，统一超参数。
- 三种 token importance 定义：
  - attention：最后一层 `[CLS]` 对非特殊 token 的平均注意力；
  - `gradient × input`：对原预测类别 logit 的绝对 attribution；
  - `LOO`：逐 token 删除后原预测类别概率的下降量。
- 评估方式：
  - 排序相似性：Spearman correlation；
  - 删除实验：confidence drop、flip rate、becomes incorrect rate。
- 说明 notebook-driven workflow 和 artifact layout，强调可复现性。

建议图：

- `Fig. 1` 实验流程图
  - 位置：Methods 开头
  - 内容：训练 DistilBERT -> 生成验证集预测 -> 计算三种 token 排序 -> 做排序相似性分析与删除实验 -> 输出定量和定性结果
  - 对应函数：`plot_experiment_workflow()`

## 5. Results

### 5.1 Base Model Performance And Seed Stability

内容规划：

- 简要报告四个 seed 的 validation accuracy、F1、loss。
- 说明模型性能稳定，后续 interpretability 结论不是由某个异常 seed 驱动。
- 可以在正文里用一张表或一张图加一句均值总结。

建议图：

- `Fig. 2` 多 seed 验证指标总览
  - 内容：accuracy / F1 / loss 三个子图，展示各 seed 表现及均值参考线
  - 对应函数：`plot_validation_metrics_overview()`

### 5.2 Agreement Between Ranking Methods

内容规划：

- 报告 attention 与 `gradient × input`、`LOO` 的平均 Spearman 相关性。
- 强调 attention 与其他方法一致性较弱。
- 同时指出 `gradient × input` 与 `LOO` 的一致性也不高，说明“重要性”本身依赖定义。

建议图：

- `Fig. 3` 排序相似性热力图
  - 内容：三种排序方法两两之间的平均 Spearman correlation
  - 对应函数：`plot_ranking_similarity_heatmap()`

### 5.3 Token Deletion Experiments

内容规划：

- 以 top-k 删除为主线分析不同方法删除 token 后的效果。
- 主结果建议围绕两件事展开：
  - confidence drop 谁最大；
  - flip rate 谁最高。
- 写作重点：
  - `LOO` 一直最强；
  - attention 比随机删除更有效；
  - 在当前设定下，attention 往往比 `gradient × input` 更强，但仍不等于“attention 就可靠”。

建议图：

- `Fig. 4` top-k 删除下的 confidence drop 曲线
  - 对应函数：`plot_deletion_metric_curve(..., metric_name="mean_confidence_drop", setting_type="top_k")`
- `Fig. 5` top-k 删除下的 flip rate 曲线
  - 对应函数：`plot_deletion_metric_curve(..., metric_name="flip_rate", setting_type="top_k")`

### 5.4 Qualitative Case Study

内容规划：

- 选择 1 到 2 个代表性句子。
- 展示三种方法给出的高排名 token 是否直观、是否影响预测。
- 优先挑选 attention 明显关注标点或功能词，而 `LOO` 更接近语义关键词的例子。
- 结合删除后的预测变化解释 why attention can be partially useful but incomplete。

建议图：

- `Fig. 6` 单个案例的 token 排名可视化
  - 内容：同一句子在 attention / `gradient × input` / `LOO` 下的 top token 与得分
  - 对应函数：`plot_qualitative_case_rankings()`

## 6. Discussion

内容规划：

- 解释 attention 为什么会包含一定信号，但不能稳定代表 token importance。
- 讨论为什么 `LOO` 更强：它直接度量删除 token 后的预测变化。
- 讨论为什么 `gradient × input` 在当前实验里没有系统性超过 attention。
- 强调局限性：
  - 单任务：只做 SST-2；
  - 单模型族：DistilBERT；
  - token 粒度是 wordpiece；
  - `LOO` 仍然是局部扰动近似，而非真正因果解释。

建议图：

- 无。Discussion 以解释和局限性为主。

## 7. Conclusion

内容规划：

- 回答研究问题：attention 不能单独作为可靠的 token importance 代理。
- 总结核心实证发现：
  - attention 与其他重要性定义只有弱相关；
  - `LOO` 删除最能破坏模型预测；
  - attention 强于随机，但不足以替代更直接的重要性度量。
- 给出后续工作：
  - 扩展到更多任务与模型；
  - 比较更多 attribution 方法；
  - 研究不同层、不同头的 attention。

建议图：

- 无。

## Appendix Plan

内容规划：

- 更多 ratio-based 删除曲线。
- 更多 qualitative cases。
- 每个 seed 的详细表格与补充统计量。

建议图：

- `Appendix Fig. A1` ratio 删除下的 confidence drop 曲线
  - 对应函数：`plot_deletion_metric_curve(..., metric_name="mean_confidence_drop", setting_type="ratio")`
- `Appendix Fig. A2` 如有需要，可补充 ratio 删除下的 flip rate 曲线
  - 同样使用 `plot_deletion_metric_curve()`
