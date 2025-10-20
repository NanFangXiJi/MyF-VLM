# 项目结构

本项目使用uv管理，可利用[`pyproject.toml`](./pyproject.toml)安装环境。


[`data/`](./data)存放数据，其包括：

- [`data/coco/`](./data/coco)，是coco数据集。此文件夹下有[`annotations`](./data/coco/annotations)、[`train2017`](./data/coco/train2017)、[`val2017`](./data/coco/val2017)，均为coco数据集。为避免文件过大，数据集图片被移除出提交文件中了。
- [`data/coco/annotations/instances_train2017_filtered.json`](./data/coco/annotations/instances_train2017_filtered.json) 是经过处理的训练集对应json
- [`data/coco/annotations/instances_val2017_filtered.json`](./data/coco/annotations/instances_val2017_filtered.json) 是经过处理的测试集对应json
- [`data/my_image/`](./data/my_image)包含我自己的图片。

[`docs/`](./docs)用于存放报告相关内容。[`/docs/report.md`](./docs/report.md)是我的报告位置。

[`model/`](./model)用于存放其他模型的预训练参数以及我的微调训练结果。为了避免文件过大，预训练参数已被移除出提交文件，但我的微调结果不包含冻结参数。为了加载我的模型，有必要使用这些预训练参数。

- [`model/FVLM_final.pth`](./model/FVLM_final.pth) 是我微调后的方案二参数(非冻结部分)。

[`output/`](./output)用于保存代码输出。这里包含我的测试输出。我的报告中会具体解释输出的内容。

[`src/`](./src)用于存放我的源代码。其中大部分代码仅临时用到，但也包含一些必要的过程，如数据集的处理等，未经过整理，且在最终提交时已不能确保可以直接运行，因此不介绍。以下列出整理后用于提交的代码

- [`src/Solution1.py`](./src/Solution1.py) 包含方案一的实现。其也包含推理代码的调用示例。
- [`src/Solution2.py`](./src/Solution2.py) 包含方案二的实现。比较长，因此不与其它代码合并。
- [`src/Solution2_inference.py`](./src/Solution2_inference.py) 包含方案二的COCO测试示例。
- [`src/Solution2_infer_one.py`](./src/Solution2_infer_one.py) 包含方案二针对单张图片 (非数据集) 的推理示例。
- [`src/Solution2_train.py`](./src/Solution2_train.py) 包含方案二的训练示例。
- [`src/Solution2_const.py`](./src/Solution2_const.py) 包含方案二用到的常量。
- [`src/evaluate.py`](./src/evaluate.py) 包含对实例检测、实例分割的性能统计。
- [`src/evaluate2.py`](./src/evaluate2.py) 包含对分类的性能统计。
