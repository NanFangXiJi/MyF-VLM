# Overview

本次任务是利用Mask R-CNN 与 CLIP 两个模型完成零样本开放字典实例检测与分割任务。

---

**Mask R-CNN** 是 R-CNN 系列模型的扩展，用于解决实例检测与分割任务。

过去R-CNN系列模型(R-CNN, Fast R-CNN, Faster R-CNN)主要用于解决目标检测任务。它们以多种方式给出大量锚框，对锚框进行筛选与预测。

Mask R-CNN 在 Faster R-CNN 的基础上增加了像素级分割的分支，能够为每个实例生成一个像素级掩码。它的像素级分割不依赖检测结果，可以与检测分支并行进行。

Mask R-CNN 是一个小模型，计算负载小，推理速度很快，但是不具备 Zero-Shot 能力。

---

**CLIP** 是由文本 Encoder 与图像 Encoder 两部分构成的模型，用于获得文本与图像的对齐的 Embeddings。

CLIP 这项工作完成于 ViT 同期。CLIP 利用对比学习的方式将文本 Encoder 与图像 Encoder 输出的 Embeddings 中，对应的靠近，不对应的则远离。这意味着可以通过计算文本与图像 Embeddings 的余弦相似度，进而判断文本与图像的对应关系。

CLIP 的一大卖点在于其 Zero-Shot 能力。它对图像的理解能力不再局限于固定的类别，而是能够判断图像与一段描述的接近程度。

CLIP 的模型规模较大，这导致其计算负载较大，推理速度较慢。但是这个模型规模为其带来了强大的泛化能力。

## Task 1

Task 1 的要求是

