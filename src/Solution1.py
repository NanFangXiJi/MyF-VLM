import pickle
import os
import json
import numpy as np
import cv2

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.layers import cat
from detectron2.modeling.roi_heads.fast_rcnn import fast_rcnn_inference
from detectron2.utils.visualizer import Visualizer
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog
import detectron2.data.transforms as T_
from detectron2.evaluation import COCOEvaluator

from transformers import CLIPProcessor, CLIPModel

test_mode = True  # 测试代码正确性

val_json = "../data/COCO/annotations/instances_val2017.json"
val_images = "../data/COCO/val2017"

val_small_json = "./val_small.json"

MRCNN_PATH = "../model/model_final_f10217.pkl"
CONFIG_FILE = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"

CLIP_PATH = "../model/clip-vit-patch32/models--openai--clip-vit-base-patch32/snapshots/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
MyMRCNN_PATH = "../model/my_mask_rcnn.pkl"
MyMRCNN_CLSFREE_PATH = "../model/my_clsfree_mask_rcnn.pkl"

class MyMaskRCNN(nn.Module):
    def __init__(self, cfg=None, original_model=None):
        """
        若提供 original_model：直接复制结构并去掉分类头；
        若提供 cfg：根据配置文件构造同结构模型，再去掉分类头；
        """
        super().__init__()
        if original_model is not None:
            self.model = original_model
        elif cfg is not None:
            self.model = build_model(cfg)
        else:
            raise ValueError("必须提供 original_model 或 cfg 之一。")

        # 去掉不需要的分类头
        if hasattr(self.model.roi_heads, "box_predictor") and hasattr(self.model.roi_heads.box_predictor, "cls_score"):
            del self.model.roi_heads.box_predictor.cls_score
            self.model.roi_heads.box_predictor.cls_score = None

        # 应用原有的数据增强
        self.aug = T_.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )

    def save_to_pkl(self, path):
        """
        将模型参数保存为.pkl文件
        """
        state_dict = self.state_dict()
        data = {
            "model": state_dict,
            "__author__": "Task 1",
            "matching_heuristics": True
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load_from_pkl(self, path):
        """
        从.pkl文件加载模型参数
        """
        with open(path, "rb") as f:
            data = pickle.load(f)

        state_dict = data["model"] if "model" in data else data
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        # print(f"[MyMaskRCNN] 从 {path} 加载完成。")
        # if missing:
        #     print("未加载参数：", missing)
        # if unexpected:
        #     print("未使用参数：", unexpected)

    def forward(self, *args, **kwargs):
        raise self.model(*args, **kwargs)


class MyDataset(Dataset):
    def __init__(self, aug, dataset_name=None, dataset_dicts=None, meta_data=None):
        if dataset_name is None:
            if dataset_dicts is None or meta_data is None:
                raise ValueError("dataset_name为None时dataset_dicts与meta_data不能为None")
            else:
                self.dataset_dicts = dataset_dicts
                self.meta_data = meta_data
        else:
            self.dataset_dicts = DatasetCatalog.get(dataset_name)
            self.meta_data = MetadataCatalog.get(dataset_name)

        self.aug = aug

    def __len__(self):
        return len(self.dataset_dicts)

    def __getitem__(self, idx):
        d = self.dataset_dicts[idx].copy()
        # 读取 cv2 图像
        org_img = cv2.imread(d["file_name"])
        d['cv2'] = org_img
        # 这一部分来自defaults的__call__
        # 转 tensor
        img = self.aug.get_transform(org_img).apply_image(org_img)
        d['image'] = torch.as_tensor(img.astype("float32").transpose(2, 0, 1))
        return d


def build_batch_loader(aug, dataset_name=None, dataset_dicts=None, meta_data=None, mrcnn_batch_size=1, shuffle=False):
    if dataset_name is None:
        if dataset_dicts is None or meta_data is None:
            raise ValueError("dataset_name为None时dataset_dicts与meta_data不能为None")
        else:
            dataset = MyDataset(aug, dataset_dicts=dataset_dicts, meta_data=meta_data)
    else:
        dataset = MyDataset(aug, dataset_name=dataset_name)
    return DataLoader(
        dataset,
        batch_size=mrcnn_batch_size,
        shuffle=shuffle,
        collate_fn=lambda batch: batch
    )


class MyZeroShotOpenVocabularyDetector(nn.Module):
    def __init__(self, clip_path, my_mask_mrcnn_path, mrcnn_cfg, device, top_k=100, nms_thresh=0.5, score_thresh=0.05):
        super().__init__()

        self.clip_model = CLIPModel.from_pretrained(clip_path).to(device)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_path)

        self.my_mask_rcnn = MyMaskRCNN(cfg=mrcnn_cfg)
        self.my_mask_rcnn.load_from_pkl(my_mask_mrcnn_path)
        self.my_mask_rcnn = self.my_mask_rcnn.to(device)
        self.device = device

        self.my_mask_rcnn.model.roi_heads.box_predictor.test_topk_per_image = top_k
        self.my_mask_rcnn.model.roi_heads.box_predictor.test_nms_thresh = nms_thresh
        self.my_mask_rcnn.model.roi_heads.box_predictor.test_score_thresh = score_thresh

    @torch.no_grad()
    def forward(self, batch, class_embeddings, clip_batch_size):

        images_mrcnn = self.my_mask_rcnn.model.preprocess_image(batch)

        features = self.my_mask_rcnn.model.backbone(images_mrcnn.tensor)
        proposals, _ = self.my_mask_rcnn.model.proposal_generator(images_mrcnn, features)

        orig_sizes = [(d["height"], d["width"]) for d in batch]
        proc_sizes = [x.shape[-2:] for x in images_mrcnn.tensor]

        boxes = [p.proposal_boxes.tensor for p in proposals]
        cropped_images = self.images_crop([d["cv2"] for d in batch], boxes, orig_sizes, proc_sizes)

        scores = self.clip_cls_pred(cropped_images, class_embeddings, clip_batch_size)
        image_shapes = [x.image_size for x in proposals]

        boxes = self.roi_heads_boxes(features, proposals)

        pred_instances, _ = fast_rcnn_inference(
            boxes,
            tuple(scores.view(len(boxes), boxes[0].shape[0], -1)),
            image_shapes,
            self.my_mask_rcnn.model.roi_heads.box_predictor.test_score_thresh,
            self.my_mask_rcnn.model.roi_heads.box_predictor.test_nms_thresh,
            self.my_mask_rcnn.model.roi_heads.box_predictor.test_topk_per_image,
        )

        pred_instances = self.my_mask_rcnn.model.roi_heads.forward_with_given_boxes(features, pred_instances)
        pred_instances = self.my_mask_rcnn.model._postprocess(pred_instances, batch, images_mrcnn.image_sizes)
        return pred_instances

    @torch.no_grad()
    def predict(self, dataset_name, mrcnn_batch_size=1, clip_batch_size=1, visualize=False, visualize_path=None):

        dataset_dicts = DatasetCatalog.get(dataset_name)
        meta_data = MetadataCatalog.get(dataset_name)

        loader = build_batch_loader(self.my_mask_rcnn.aug, dataset_dicts=dataset_dicts, meta_data=meta_data,
                                    mrcnn_batch_size=mrcnn_batch_size, shuffle=False)

        results = []

        class_name_list = self.class_name_list_prepare(meta_data.thing_classes)
        class_embeddings = self.get_cls_embedding(class_name_list)

        for batch in loader:
            batch_results = []
            pred_instances = self.forward(batch, class_embeddings, clip_batch_size)

            # 后处理，按照detectron2的格式
            for det, inp in zip(pred_instances, batch):
                det["instances"] = det["instances"].to("cpu")
                out_dict = {
                    "image_id": inp["image_id"],
                    "instance": det['instances'],
                    "image": inp["image"].cpu(),
                    "cv2": inp["cv2"],
                    "file_name": inp["file_name"]
                }
                batch_results.append(out_dict)
                results.append(out_dict)

            # 清理一下内存
            del pred_instances, batch
            torch.cuda.empty_cache()

            if visualize:
                self.visualize(batch_results, visualize_path, meta_data)

        return results

    def visualize(self, results, visualize_path, meta_data):
        for result in results:
            img = result["cv2"][:, :, ::-1]

            # 创建 Visualizer
            v = Visualizer(img, metadata=meta_data, scale=1.2)

            out = v.draw_instance_predictions(result["instance"])
            img_vis = out.get_image()

            img_path = os.path.join(visualize_path, f"{os.path.basename(result['file_name'])}")
            if not cv2.imwrite(img_path, img_vis[..., ::-1]):  # 转回 BGR
                raise IOError(f"Failed to visualize image {img_path}")
            print(f"saved at {img_path}")

    def roi_heads_boxes(self, features, proposals):
        # 这一段改写自detectron2.modeling.roi_heads.roi_heads.StandardROIHeads._forward_box及其调用的函数
        # 跳过了框的筛选等部分，这一部分会在CLIP预测类别后进行
        features = [features[f] for f in self.my_mask_rcnn.model.roi_heads.box_in_features]
        box_features = self.my_mask_rcnn.model.roi_heads.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.my_mask_rcnn.model.roi_heads.box_head(box_features)
        if box_features.dim() > 2:
            box_features = torch.flatten(box_features, start_dim=1)
        proposal_deltas = self.my_mask_rcnn.model.roi_heads.box_predictor.bbox_pred(box_features)
        del box_features

        num_prop_per_image = [len(p) for p in proposals]
        proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)
        predict_boxes = self.my_mask_rcnn.model.roi_heads.box_predictor.box2box_transform.apply_deltas(
            proposal_deltas,
            proposal_boxes,
        )  # Nx(KxB)
        return predict_boxes.split(num_prop_per_image)

    @staticmethod
    def images_crop(images, boxes, orig_sizes, proc_sizes):
        boxes_on_original = []
        for boxes_per_image, (proc_h, proc_w), (orig_h, orig_w) in zip(boxes, proc_sizes, orig_sizes):
            scale_x = orig_w / proc_w
            scale_y = orig_h / proc_h
            boxes_scaled = boxes_per_image.clone()
            boxes_scaled[:, 0::2] *= scale_x
            boxes_scaled[:, 1::2] *= scale_y
            boxes_on_original.append(boxes_scaled)

        cropped_images = []

        for img, boxes_per_image in zip(images, boxes_on_original):
            box_reshape = boxes_per_image.reshape(-1, 4)
            h_img, w_img = img.shape[:2]

            for box in box_reshape:
                x1, y1, x2, y2 = box.tolist()

                # 计算宽高
                w = x2 - x1
                h = y2 - y1

                # 如果太小，则调整到最小尺寸
                if w < 3:
                    delta = (3 - w) / 2
                    x1 = max(0, x1 - delta)
                    x2 = min(w_img, x2 + delta)
                if h < 3:
                    delta = (3 - h) / 2
                    y1 = max(0, y1 - delta)
                    y2 = min(h_img, y2 + delta)

                # 转为 int 并截断边界
                x1 = int(np.clip(x1, 0, w_img - 1))
                x2 = int(np.clip(x2, 0, w_img))
                y1 = int(np.clip(y1, 0, h_img - 1))
                y2 = int(np.clip(y2, 0, h_img))

                # OpenCV 切片裁剪 ([y1:y2, x1:x2])
                crop = img[y1:y2, x1:x2]

                # 转换为 CHW 格式
                cropped_images.append(np.transpose(crop, (2, 0, 1)))

        return cropped_images

    def clip_cls_pred(self, images, cls_embeddings, clip_batch_size):
        img_embeddings_list = []

        for j in range(0, len(images), clip_batch_size):
            batch_imgs = images[j:j + clip_batch_size]
            inputs = self.clip_processor(images=batch_imgs, return_tensors='pt', padding=True).to(self.device)
            with torch.no_grad():
                img_embeddings = self.clip_model.get_image_features(**inputs)
                img_embeddings_list.append(img_embeddings)
        img_embeddings = torch.cat(img_embeddings_list, dim=0)
        similarity = img_embeddings @ cls_embeddings.T
        probs = similarity.softmax(dim=-1)
        return probs

    def class_name_list_prepare(self, class_name_list):
        class_name_list = ["a photo of " + cls for cls in class_name_list]
        class_name_list.append("a random photo with no specific object")
        return class_name_list

    def get_cls_embedding(self, class_name_list):
        class_inputs = self.clip_processor(text=class_name_list, return_tensors="pt", padding=True).to(
            self.clip_model.device)
        return self.clip_model.get_text_features(**class_inputs).to(self.clip_model.device)

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(CONFIG_FILE))
cfg.MODEL.WEIGHTS = MRCNN_PATH
cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG = True
cfg.MODEL.DEVICE = "cuda"

model = MyZeroShotOpenVocabularyDetector(CLIP_PATH, MyMRCNN_CLSFREE_PATH, cfg, torch.device("cuda"), nms_thresh=0.2, score_thresh=0.5)
model.eval()

register_coco_instances("coco_val", {}, val_json, val_images)


def test(model: MyZeroShotOpenVocabularyDetector, dataset_name=None, output_dir=None, mrcnn_batch_size=1,
         clip_batch_size=1):
    dataset_dicts = DatasetCatalog.get(dataset_name)
    meta_data = MetadataCatalog.get(dataset_name)

    loader = build_batch_loader(model.my_mask_rcnn.aug, dataset_dicts=dataset_dicts, meta_data=meta_data,
                                mrcnn_batch_size=mrcnn_batch_size, shuffle=False)

    evaluator = COCOEvaluator(dataset_name if dataset_name else "custom_coco", output_dir=output_dir)
    evaluator.reset()

    class_name_list = model.class_name_list_prepare(meta_data.thing_classes)
    class_embeddings = model.get_cls_embedding(class_name_list)

    for batch in loader:
        with torch.no_grad():
            pred_instances = model.forward(batch, class_embeddings=class_embeddings, clip_batch_size=clip_batch_size)

        # 转换为 Detectron2 标准格式
        for det, inp in zip(pred_instances, batch):
            det["instances"] = det["instances"].to("cpu")
            out_dict = {
                "image_id": inp["image_id"],
                "instances": det["instances"],
                "height": inp["height"],
                "width": inp["width"]
            }

            evaluator.process([inp], [out_dict])

    metrics = evaluator.evaluate()

    return metrics


@torch.no_grad()
def pred_one(model, image_path, class_name_list, clip_batch_size=1, visualize=False, visualize_path=None):
    model.eval()

    d = {}
    d["file_name"] = image_path
    org_img = cv2.imread(d["file_name"])
    d['cv2'] = org_img
    # 这一部分来自defaults的__call__
    # 转 tensor
    img = model.my_mask_rcnn.aug.get_transform(org_img).apply_image(org_img)
    d['image'] = torch.as_tensor(img.astype("float32").transpose(2, 0, 1))
    d['height'] = d['cv2'].shape[0]
    d['width'] = d['cv2'].shape[1]

    class_name_list_long = model.class_name_list_prepare(class_name_list)
    class_embeddings = model.get_cls_embedding(class_name_list_long)

    pred_instances = model.forward([d], class_embeddings, clip_batch_size)

    pred_instances[0]['instances'] = pred_instances[0]['instances'].to("cpu")
    out_dict = {
        "instance": pred_instances[0]['instances'],
        "image": d["image"].cpu(),
        "cv2": d["cv2"],
        "file_name": d["file_name"]
    }

    if visualize:
        model.visualize([out_dict], visualize_path, {"thing_classes": class_name_list})


# model.predict("coco_val", 5, 1000, True, "../output/Solution1_res")  # 推理示例

# test(model, "coco_val", output_dir="../output/test", mrcnn_batch_size=2, clip_batch_size=1000)  # 测试示例

# 单张图片推理示例
image_path = "../data/my_image/my_books.jpg"
class_name_list = ["box", "lamp", "keyboard", "mouse", "laptop", "headset", "bag", "book", "chair", "toll", "clothing", "umbrella", "coat hanger"]
pred_one(model, image_path, class_name_list, 1000, True, "../output/my_image_test_sol1")
