# Project Abstract

## Title

Do Attention Weights Reflect Token Importance? A Small-Scale Study with DistilBERT on Sentiment Classification

## Authors

- Tianhao Gong
- CHENKAI WANG

## Keywords

Transformer Interpretability, Attention Mechanisms, Token Importance, Model Explanation, Sentiment Classification, DistilBERT

## Abstract (English)

Attention weights are often treated as a simple explanation of which words matter most in a Transformer's prediction. However, prior work suggests that attention may not always reflect true token importance. In this project, we investigate this question in a focused and manageable setting using DistilBERT on the SST-2 sentiment classification task. Our motivation is to better understand whether attention can be used as a reliable interpretability signal, rather than assuming that highly attended tokens are always the most influential ones. We compare attention-based token rankings with two alternative importance measures, gradient × input and leave-one-out (LOO), where LOO is applied only to a sampled subset due to its computational cost. We then evaluate their relationship through ranking similarity and token deletion experiments, examining how prediction confidence and output labels change after removing top-ranked tokens. We expect that attention will show limited agreement with these alternative importance measures, and that removing tokens identified by them will affect model predictions more than removing tokens selected by attention alone. This project aims to provide a focused empirical assessment of when attention can serve as a useful explanation in Transformer-based text classification.

## 摘要（中文）

注意力权重常被视为一种简单的解释方式，用来说明在 Transformer 的预测中哪些词最重要。然而，已有研究表明，注意力并不总能真实反映 token 的实际重要性。在本项目中，我们在一个聚焦且可控的设置下研究这一问题，具体使用 DistilBERT 在 SST-2 情感分类任务上开展实验。我们的动机是进一步理解：注意力是否可以作为一种可靠的可解释性信号，而不是默认认为获得高注意力的 token 一定就是最有影响力的部分。

我们将基于注意力的 token 排序，与两种替代性的重要性度量进行比较：`gradient × input` 和 `leave-one-out (LOO)`。由于 LOO 的计算成本较高，我们仅在抽样得到的子集上应用该方法。随后，我们通过排序相似性分析和 token 删除实验来评估这些度量之间的关系，考察删除高排名 token 后，模型预测置信度以及输出标签会如何变化。

我们预期，注意力与这些替代性重要性度量之间的一致性会较为有限；同时，由这些替代方法识别出的 token，在被移除后，对模型预测产生的影响将大于仅依据注意力所选择的 token。本项目旨在对这样一个问题给出聚焦的实证评估：在基于 Transformer 的文本分类中，注意力究竟在什么条件下可以作为一种有用的解释。

## OpenReview Metadata

- Email Sharing: We authorize the sharing of all author emails with Program Chairs.
- Data Release: We authorize the release of our submission and author names to the public in the event of acceptance.
