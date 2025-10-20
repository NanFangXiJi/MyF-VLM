# Overview

本次任务是利用Mask R-CNN 与 CLIP 两个模型完成零样本开放字典实例检测与分割任务。

---

# 基础模型介绍

## Mask R-CNN

![](./Mask_RCNN.png)

**Mask R-CNN** 是 R-CNN 系列模型的扩展，用于解决实例检测与分割任务。

过去R-CNN系列模型(R-CNN, Fast R-CNN, Faster R-CNN)主要用于解决目标检测任务。它们以多种方式给出大量锚框，对锚框进行筛选与预测。

Mask R-CNN 在 Faster R-CNN 的基础上增加了像素级分割的分支，能够为每个实例生成一个像素级掩码。它的像素级分割不依赖检测结果，可以与检测分支并行进行。

Mask R-CNN 是一个小模型，计算负载小，推理速度很快，但是不具备 Zero-Shot 能力。

Mask R-CNN 首先通过 backbone 提取多层次特征。这一特征首先送入 RPN。RPN 会给出区域提议并初步判断其是否为背景。此后 RPN 的区域提议与 FPN 的多层次特征会被一起送入 ROI Align。ROI Align 负责给出各提议区域的特征。这些区域特征会交给 Bounding Box Head 和 Mask Head。Bounding Box Head 会预测各个框的类别并对框给出偏移量，Mask Head 会给出物体的对应掩码。 

Mask Head 对每个框默认会针对全部类别都输出一个预测，但也可以设置为类别不可知模式，此时每个框只会有一个预测。

---

## CLIP

![](./CLIP.png)

**CLIP** 是由文本 Encoder 与图像 Encoder 两部分构成的模型，用于获得文本与图像的对齐的 Embeddings。

CLIP 这项工作完成于 ViT 同期。CLIP 利用对比学习的方式将文本 Encoder 与图像 Encoder 输出的 Embeddings 中，对应的靠近，不对应的则远离。这意味着可以通过计算文本与图像 Embeddings 的余弦相似度，进而判断文本与图像的对应关系。

CLIP 的一大优势在于其 Zero-Shot 能力。它对图像的理解能力不再局限于固定的类别，而是能够判断图像与一段描述的接近程度。

CLIP 的模型规模较大，这导致其计算负载较大，推理速度较慢。但是这个模型规模为其带来了强大的泛化能力。

CLIP 的图像编码器在 F-VLM 中被描述为有两部分，分别是特征提取与池化。图中已经拆分体现。

# 我的解决方案

## 方案一：获取区域提案后使用 CLIP 处理裁剪图片以预测

![](./MyMethod1.png)

我的第一个方案是，获取区域提案后使用 CLIP 处理裁剪图片以预测。

首先去掉 Mask R-CNN 的类型预测分支。

对于所有的类别，直接使用 CLIP 的 Text Encoder 处理其类名(或一段描述，如`a photo of a cat`)，获得类别对应的 Embedding。

在 FPN，RPN 后获得区域提议。接着依据此区域提议，对原始图片进行裁切，并将裁切得到的图像交给 CLIP 计算裁切图像的 Embedding。这里，裁切区域选择的不是经过 Bounding Box Head 调整后的，因为此输出在类别可知模式会让 CLIP 需要处理的图片数量乘上其检测的类别个数，极大地增加了计算量。

两个 Embedding 计算余弦相似度后，即可得到各个类别对应的分数。其余的`box regression`与`mask prediction`仍然沿用之前的计算方法。

### 方案一的优势

1. 此方法的大多数计算都是沿用之前的方法，代码比较好写。
2. 此方法不需要任何的进一步训练，可以直接应用预训练的模型，也能获得比较好的效果。

### 方案一的劣势
1. 此方法计算量巨大。每张图片经过 RPN 生成的`box proposal`有1000之巨，这意味着对于每张图片要进行1000次 CLIP 的推理。CLIP 本身的计算量巨大，这导致计算速度难以接受。
2. 此方法会造成严重的显存浪费。在CLIP计算阶段，其显存占用产生远高于其它阶段的峰值。为了弥补此劣势，我使用`CLIP_batch`，以限制同时推理 CLIP 的图片数量，并确保 Mask R-CNN 上能有更多图片的并行。

## 方案二：利用 F-VLM 的思想构建
![](./MyMethod2.png)

此方案使用了 F-VLM 的思想来设计架构。

最终的输出仍然还是box、mask与class score，为方便查看，其边框均被加粗。

本架构要求 CLIP 相关参数被冻结，不进行训练。因此架构设计将让不产生 loss 的路径避免出现不属于 CLIP 的需要训练得到参数的部件。

### 推理过程

其将 CLIP 的 Image Encoder 分为两阶段。第一阶段将产生一个形状为`(B,d^2+1,C)`的image feature。对于第二维，$d$是 ViT 对图像的分割($d\times d$)，该维其余一位是CLS。我的设计将CLS抛弃，其余部分转换为形状为`(B,C,d,d)`的张量，即可将其视作图像的一种 feature。

为了让图像 feature 能够输入 FPN，我添加了 CLIPtoFPN Adapter，将其进行多规格的上采样，代替了原有的 `bottom_up` 步骤。此后的步骤与 Mask R-CNN 完全一致，经过 RPN 提议后交给 ROI Align，由 Mask Head 生成 masks，由 Bounding Box Head 生成调整后的 Boxes。

此时，Bounding Box Head 输出中间会产生一个 box feature，我为其添加了一个 Projection Adapter 来获取与 CLIP 对齐的 embeddings。此 embeddings 将会与 text embeddings 求余弦相似度，生成 `class score 1`。

另外，Image Encoder Stage 1生成的图像 feature 可以与 Bounding Box Head 输出的 Boxes 经 ROI Align 后成为各个 Boxes 的 feature，再由 CLIP 的 Image Encoder Stage 2 处理成为 embeddings，与文本 embeddings 求余弦相似度即可得到 `class score 2`。

最终的 class score 由 `class score 1` 与 `class score 2` 求几何平均值得到。

### 训练过程

对于其训练过程，其 box regression loss 与 mask loss 都与 Mask R-CNN 的原始损失求解方式一致。

由于 class score 2 的计算路径不涉及任何需要训练的参数，因此这一段路径不会在训练中被计算。

在计算 class score 1 时，进行了 aligned box embeddings 与 text embeddings 的余弦损失计算。在训练期间，会计算这二者的对比损失。

> 这里有必要提到CLIP的阶段分配方案的设计细节与解释。
> 
> F-VLM 认为，CLIP 的两阶段分别为特征提取与池化两部分。这也是我最初尝试的设计。然而这种设计在实现中的效果并不好。其给出的 `score 2` 的效果很糟糕。
> 
> 我会尝试解释这种糟糕的原因。我的原架构将第一阶段的输出中的 CLS 去掉，利用其 patch 来还原出图像特征。然而在原 CLIP 中，这里输出的特征是用于直接交给池化层的，而交给池化层的部分只包含 CLS 部分，而不包含 patch 部分。这可能意味着除了 CLS 部分以外的部分的特征代表性是难以保证的。
> 
> 然而，CLIP 的特征提取部分也有自身的阶段，在代码中分为 `embeddings` 阶段与 `encoder` 阶段。在 `embeddings` 阶段输出的特征具有完全一致的、便利的形状，而且由于其输出仍然属于较早期的特征，因此可以认为其信息在对应位置仍然比较完整。经过实验，对这里输出的特征直接在框上做 ROI Align ，可以很好地得到局部的特征。再将此局部特征与 CLS 合并交给`encoder`部分以及池化部分，生成的 `score 2` 不需要额外的训练(也没有可训练的参数)，就能得到不错的效果。
> 
> 不过，此设计有一个缺陷。如果按照原架构利用最后的特征，其经过 ROI Align 后只需要对每个框生成一个特征向量，因为它将会直接被送入池化层，而池化层只需要接受一个特征向量。然而如果利用中间特征，由于其仍然需要交给 `encoder` 作为输入，因此仍然需要保持原形状，也就是针对每一个patch都要生成一个特征向量，这会导致输出量在此时暴增(因为此时第一维不再是batch_size，而是所有图片候选框的个数，这个数字已经很大，乘上patch的个数，通常是$7\times 7$，会超过处理能力)。通常，在这一步如果强行并行，会导致出现显存峰值。为了避免此情况，我只能使用循环以batch_size为单位计算。这为推理带来了性能瓶颈。
> 
> 在图片中，我仍然将CLIP分为特征提取与池化两阶段，不过 `encoder` 这一部分归于池化层部分了。


