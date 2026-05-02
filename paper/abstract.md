# Project Abstract

## Title

Do Attention Weights Reflect Token Importance? A Small-Scale Study with DistilBERT on Sentiment Classification

## Authors

- Tianhao Gong
- CHENKAI WANG

## Keywords

Transformer Interpretability, Attention Mechanisms, Token Importance, Model Explanation, Sentiment Classification, DistilBERT

## Abstract (English)

Attention weights are often used as a convenient explanation of which words matter most in a Transformer prediction, yet prior work has questioned whether they reflect true token importance. In this project, we revisit that claim in a controlled setting by fine-tuning DistilBERT on the SST-2 sentiment classification task and repeating the experiment across four random seeds. For each validation example, we rank tokens with three signals: last-layer `[CLS]` attention, gradient × input, and leave-one-out (LOO) deletion. We then compare these rankings with Spearman correlation and evaluate them through token-deletion tests on the full validation set, measuring confidence drops and prediction flips after removing top-ranked tokens. Our results show that attention has only weak agreement with the alternative rankings, with mean Spearman correlations of about 0.22 against gradient × input and 0.18 against LOO. Deletion experiments further show that LOO-selected tokens produce the largest confidence drops and the highest flip rates, while attention-based deletions are stronger than random deletions but do not match LOO. Overall, our findings suggest that attention captures some useful signal but is not a reliable proxy for token importance on its own. This study provides a compact empirical benchmark and a reproducible workflow for examining interpretability claims in Transformer-based text classification.

## 摘要（中文）

注意力权重常被视为一种简单的解释方式，用来说明在 Transformer 的预测中哪些词最重要。然而，已有研究表明，注意力并不总能真实反映 token 的实际重要性。在本项目中，我们在一个聚焦且可控的设置下研究这一问题，具体使用 DistilBERT 在 SST-2 情感分类任务上开展实验。我们的动机是进一步理解：注意力是否可以作为一种可靠的可解释性信号，而不是默认认为获得高注意力的 token 一定就是最有影响力的部分。

我们将基于注意力的 token 排序，与两种替代性的重要性度量进行比较：`gradient × input` 和 `leave-one-out (LOO)`。我们在 SST-2 上微调 DistilBERT，并在四个随机种子下重复实验；随后对完整验证集中的每个样本计算三种排序，通过 Spearman 相关系数比较它们的一致性，并通过 token 删除实验考察删除高排名 token 后模型置信度和预测标签的变化。

实验结果表明，注意力与替代方法之间的一致性较弱：注意力与 `gradient × input` 的平均 Spearman 相关系数约为 0.22，与 `LOO` 的平均相关系数约为 0.18。删除实验进一步显示，`LOO` 选出的 token 会带来最大的置信度下降和最高的预测翻转率；基于注意力的删除虽然显著强于随机删除，但整体上仍弱于 `LOO`。这些结果说明，注意力可以反映一部分有用信号，但本身并不足以作为稳定可靠的 token 重要性代理。本项目因此提供了一个聚焦、可复现的实证基准，用于检验 Transformer 文本分类中关于可解释性的常见主张。

## OpenReview Metadata

- Email Sharing: We authorize the sharing of all author emails with Program Chairs.
- Data Release: We authorize the release of our submission and author names to the public in the event of acceptance.
