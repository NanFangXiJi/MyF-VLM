import cv2
import math
import random
import numpy as np
import os
from pycocotools import mask as mask_utils
import json
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda.amp import autocast, GradScaler

from transformers.models.clip.modeling_clip import CLIPVisionTransformer
from transformers import CLIPProcessor, CLIPModel

from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
from detectron2.modeling.proposal_generator.rpn import RPN
from detectron2.modeling.proposal_generator.proposal_utils import add_ground_truth_to_proposals
from detectron2.modeling.roi_heads.roi_heads import StandardROIHeads
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers, fast_rcnn_inference
from detectron2.modeling.roi_heads.mask_head import MaskRCNNConvUpsampleHead
from detectron2.modeling.box_regression import _dense_box_regression_loss
from detectron2.modeling.poolers import convert_boxes_to_pooler_format
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import Boxes, Instances, BitMasks
from detectron2.structures.boxes import pairwise_iou
from detectron2.data.datasets import register_coco_instances
from detectron2.layers import ROIAlign
from detectron2.evaluation import COCOEvaluator
from detectron2.utils.visualizer import Visualizer


def cosine_scheduler_with_warmup(optimizer, warmup_steps, total_steps, min_lr_ratio=0.1):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = (current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return min_lr_ratio + (1 - min_lr_ratio) * cosine_decay
    return LambdaLR(optimizer, lr_lambda)


def init_weights_kaiming(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def init_weights_xavier(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def contrastive_loss(scores, gt_classes, temperature=0.07):
    # softmax 温度缩放
    logits = scores / temperature

    # 使用交叉熵作为对比损失
    with autocast(enabled=False):
        loss = F.cross_entropy(logits.float(), gt_classes.long())
    return loss


class CLIPVisionTransformerSplit(CLIPVisionTransformer):
    def forward_features(self, pixel_values):
        """对应 feature extractor 部分"""
        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.pre_layrnorm(hidden_states)
        return hidden_states

    def forward_pool(self, hidden_states):
        """对应 last feature pooling layer 部分，这一部分的输出不能直接用，还需要 visual projection 来投影"""
        encoder_outputs = self.encoder(inputs_embeds=hidden_states)
        last_hidden_state = encoder_outputs[0][:, 0, :]
        pooled_output = self.post_layernorm(last_hidden_state)
        return pooled_output


class TwoStageCLIPModel(CLIPModel):
    """
    拓展版 CLIP 模型，支持显式分离视觉编码的两个阶段：
    1. Feature Extractor (patch embedding + transformer encoder)
    2. Last Feature Pooling Layer (CLS pooling + LayerNorm)
    """

    def __init__(self, config):
        super().__init__(config)
        self.vision_model = CLIPVisionTransformerSplit(config.vision_config)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

        vision_config = model.config.vision_config
        new_vision_model = CLIPVisionTransformerSplit(vision_config)
        new_vision_model.load_state_dict(model.vision_model.state_dict())
        model.vision_model = new_vision_model

        return model

    def get_image_features_stage1(self, pixel_values: torch.FloatTensor) -> torch.FloatTensor:
        """
        获取图像的 patch-level 特征 (Feature Extractor 输出)
        对应 self.vision_model.embeddings + self.vision_model.encoder
        这里的输出中，seq_len维度的首位是CLS，其余是图像各分块的embeddings。
        """
        return self.vision_model.forward_features(pixel_values)  # shape: (batch, seq_len, hidden_dim)

    def get_image_features_stage2(self, hidden_state: torch.FloatTensor) -> torch.FloatTensor:
        """
        从 stage1 输出计算 pooled image feature
        对应 self.vision_model.post_layernorm(CLS token) + self.visual_projection(pooled_state)
        """
        pooled_state = self.vision_model.forward_pool(hidden_state)  # shape: (batch, hidden_dim)
        return self.visual_projection(pooled_state)


class MyRPN(RPN):
    def forward(self, image_sizes, features, gt_instances=None):
        """
        原本传入 image 只为了 image_size，这次直接传入image_size，避免还要image
        """
        features = [features[f] for f in self.in_features]
        anchors = self.anchor_generator(features)

        pred_objectness_logits, pred_anchor_deltas = self.rpn_head(features)
        # Transpose the Hi*Wi*A dimension to the middle:
        pred_objectness_logits = [
            # (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N, Hi*Wi*A)
            score.permute(0, 2, 3, 1).flatten(1)
            for score in pred_objectness_logits
        ]
        pred_anchor_deltas = [
            # (N, A*B, Hi, Wi) -> (N, A, B, Hi, Wi) -> (N, Hi, Wi, A, B) -> (N, Hi*Wi*A, B)
            x.view(x.shape[0], -1, self.anchor_generator.box_dim, x.shape[-2], x.shape[-1])
            .permute(0, 3, 4, 1, 2)
            .flatten(1, -2)
            for x in pred_anchor_deltas
        ]

        if self.training:
            assert gt_instances is not None, "RPN requires gt_instances in training!"
            gt_labels, gt_boxes = self.label_and_sample_anchors(anchors, gt_instances)
            losses = self.losses(
                anchors, pred_objectness_logits, gt_labels, pred_anchor_deltas, gt_boxes
            )
        else:
            losses = {}
        proposals = self.predict_proposals(
            anchors, pred_objectness_logits, pred_anchor_deltas, image_sizes
        )
        return proposals, losses

    @torch.jit.unused
    def losses(self, anchors, pred_objectness_logits, gt_labels, pred_anchor_deltas, gt_boxes):
        num_images = len(gt_labels)
        gt_labels = torch.stack(gt_labels)
        pos_mask = gt_labels == 1

        localization_loss = _dense_box_regression_loss(
            anchors,
            self.box2box_transform,
            pred_anchor_deltas,
            gt_boxes,
            pos_mask,
            box_reg_loss_type=self.box_reg_loss_type,
            smooth_l1_beta=self.smooth_l1_beta,
        )

        valid_mask = gt_labels >= 0
        objectness_loss = F.binary_cross_entropy_with_logits(
            torch.cat(pred_objectness_logits, dim=1)[valid_mask],
            gt_labels[valid_mask].to(torch.float32),
            reduction="sum",
        )
        normalizer = self.batch_size_per_image * num_images
        losses = {
            "loss_rpn_cls": objectness_loss / normalizer,
            "loss_rpn_loc": localization_loss / normalizer,
        }
        losses = {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}
        return losses


class MyBoxPredictor(FastRCNNOutputLayers):
    def forward(self, x):
        """
        Args:
            x: per-region features of shape (N, ...) for N bounding boxes to predict.

        Returns:
            (Tensor, Tensor):
            First tensor: shape (N,K+1), scores for each of the N box. Each row contains the
            scores for K object categories and 1 background class.

            Second tensor: bounding box regression deltas for each box. Shape is shape (N,Kx4),
            or (N,4) for class-agnostic regression.
        """
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        # scores = self.cls_score(x)  # scores不需要有
        proposal_deltas = self.bbox_pred(x)
        return None, proposal_deltas

    def losses(self, predictions, proposals):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were used
                to compute predictions. The fields ``proposal_boxes``, ``gt_boxes``,
                ``gt_classes`` are expected.

        Returns:
            Dict[str, Tensor]: dict of losses
        """
        scores, proposal_deltas = predictions

        # parse classification outputs
        gt_classes = torch.cat([p.gt_classes for p in proposals], dim=0)

        # parse box regression outputs
        proposal_boxes = torch.cat([p.proposal_boxes.tensor for p in proposals], dim=0)  # Nx4
        assert not proposal_boxes.requires_grad, "Proposals should not require gradients!"
        # If "gt_boxes" does not exist, the proposals must be all negative and
        # should not be included in regression loss computation.
        # Here we just use proposal_boxes as an arbitrary placeholder because its
        # value won't be used in self.box_reg_loss().
        gt_boxes = torch.cat(
            [(p.gt_boxes if p.has("gt_boxes") else p.proposal_boxes).tensor for p in proposals],
            dim=0,
        )

        losses = {
            "loss_cls": contrastive_loss(scores, gt_classes),
            "loss_box_reg": self.box_reg_loss(
                proposal_boxes, gt_boxes, proposal_deltas, gt_classes
            ),
        }
        return {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}


class MyMaskHead(MaskRCNNConvUpsampleHead):
    def forward(self, x, instances):
        """
        Args:
            x: input region feature(s) provided by :class:`ROIHeads`.
            instances (list[Instances]): contains the boxes & labels corresponding
                to the input features.
                Exact format is up to its caller to decide.
                Typically, this is the foreground instances in training, with
                "proposal_boxes" field and other gt annotations.
                In inference, it contains boxes that are already predicted.

        Returns:
            A dict of losses in training. The predicted "instances" in inference.
        """
        x = self.layers(x)
        if self.training:
            return {"loss_mask": mask_rcnn_loss(x, instances, self.vis_period) * self.loss_weight}
        else:
            mask_rcnn_inference(x, instances)
            return instances


@torch.jit.unused
def mask_rcnn_loss(pred_mask_logits: torch.Tensor, instances, vis_period: int = 0):
    """
    Compute the mask prediction loss defined in the Mask R-CNN paper.

    Args:
        pred_mask_logits (Tensor): A tensor of shape (B, C, Hmask, Wmask) or (B, 1, Hmask, Wmask)
            for class-specific or class-agnostic, where B is the total number of predicted masks
            in all images, C is the number of foreground classes, and Hmask, Wmask are the height
            and width of the mask predictions. The values are logits.
        instances (list[Instances]): A list of N Instances, where N is the number of images
            in the batch. These instances are in 1:1
            correspondence with the pred_mask_logits. The ground-truth labels (class, box, mask,
            ...) associated with each instance are stored in fields.
        vis_period (int): the period (in steps) to dump visualization.

    Returns:
        mask_loss (Tensor): A scalar tensor containing the loss.
    """
    cls_agnostic_mask = pred_mask_logits.size(1) == 1
    total_num_masks = pred_mask_logits.size(0)
    mask_side_len = pred_mask_logits.size(2)
    assert pred_mask_logits.size(2) == pred_mask_logits.size(3), "Mask prediction must be square!"

    gt_classes = []
    gt_masks = []
    for instances_per_image in instances:
        if len(instances_per_image) == 0:
            continue
        if not cls_agnostic_mask:
            gt_classes_per_image = instances_per_image.gt_classes.to(dtype=torch.int64)
            gt_classes.append(gt_classes_per_image)

        gt_masks_per_image = instances_per_image.gt_masks.crop_and_resize(
            instances_per_image.proposal_boxes.tensor, mask_side_len
        ).to(device=pred_mask_logits.device)
        # A tensor of shape (N, M, M), N=#instances in the image; M=mask_side_len
        gt_masks.append(gt_masks_per_image)

    if len(gt_masks) == 0:
        return pred_mask_logits.sum() * 0

    gt_masks = torch.cat(gt_masks, dim=0)

    if cls_agnostic_mask:
        pred_mask_logits = pred_mask_logits[:, 0]
    else:
        indices = torch.arange(total_num_masks)
        gt_classes = torch.cat(gt_classes, dim=0)
        pred_mask_logits = pred_mask_logits[indices, gt_classes]

    gt_masks = gt_masks.to(dtype=torch.float32)

    mask_loss = F.binary_cross_entropy_with_logits(pred_mask_logits, gt_masks, reduction="mean")
    return mask_loss


def mask_rcnn_inference(pred_mask_logits: torch.Tensor, pred_instances):
    cls_agnostic_mask = pred_mask_logits.size(1) == 1

    if cls_agnostic_mask:
        mask_probs_pred = pred_mask_logits.sigmoid()
    else:
        num_masks = pred_mask_logits.shape[0]
        class_pred = torch.cat([i.pred_classes for i in pred_instances])
        indices = torch.arange(num_masks, device=class_pred.device)
        mask_probs_pred = pred_mask_logits[indices, class_pred][:, None].sigmoid()

    num_boxes_per_image = [len(i) for i in pred_instances]
    mask_probs_pred = mask_probs_pred.split(num_boxes_per_image, dim=0)

    for prob, instances in zip(mask_probs_pred, pred_instances):
        instances.pred_masks = prob


class MyROIHeads(StandardROIHeads):
    def __init__(self, *, box_in_features, box_pooler, box_head, box_predictor: nn.Module, **kwargs):
        super().__init__(box_in_features=box_in_features, box_pooler=box_pooler, box_head=box_head,
                         box_predictor=box_predictor, **kwargs)
        self.projection_adapter = nn.Linear(1024, 512)  # 用于对齐的层

    def forward(self, features, proposals, text_embeddings, targets=None):
        if self.training:
            assert targets, "'targets' argument is required during training"
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        if self.training:
            losses = self._forward_box(features, proposals, text_embeddings)
            losses.update(self._forward_mask(features, proposals))
            return proposals, losses
        else:
            pred_instances, scores = self._forward_box(features, proposals, text_embeddings)

            # 接下来就是self.forward_with_given_boxes的调用，这个是用来求mask和key point的。这里只需要mask，同时必须设定为类型不可知模式，因为此时预测的类别还没有给出。
            # 其实稍加改动mask_rcnn_inderence(roi_heads\mask_head)就可以兼容类型可知模式，这里已exhausted……不管了，直接用原来的
            # pred_instances = self._forward_mask(features, pred_instances)  # 显存爆炸，分完box以后再做这一部分

            return pred_instances, scores

    def _forward_box(self, features, proposals, text_embeddings):
        features = [features[f] for f in self.box_in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)
        _, box_deltas = self.box_predictor(box_features)
        box_features = self.projection_adapter(box_features)
        box_features = F.normalize(box_features, p=2, dim=1)  # 这就是最终的 embedding 了

        scores = box_features @ text_embeddings.T
        # scores = scores.softmax(-1)

        if self.training:
            losses = self.box_predictor.losses((scores, box_deltas), proposals)
            # proposals is modified in-place below, so losses must be computed first.
            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        (scores, box_deltas), proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            return losses
        else:
            # 第一步调用boxes = self.predict_boxes(predictions, proposals)，这里把偏移边框实际应用，得到预测框，可以用
            boxes = self.box_predictor.predict_boxes((scores, box_deltas), proposals)
            for proposals_per_image, boxes_per_image in zip(proposals, boxes):
                proposals_per_image.pred_boxes = Boxes(boxes_per_image)

            scores = scores.softmax(-1)

            return proposals, scores  # 这里，分数也需要输出出去。proposals实际是Instances，加上了偏移后的boxes

    @torch.no_grad()
    def label_and_sample_proposals(self, proposals, targets):
        """
        Prepare some proposals to be used to train the ROI heads.
        It performs box matching between `proposals` and `targets`, and assigns
        training labels to the proposals.
        It returns ``self.batch_size_per_image`` random samples from proposals and groundtruth
        boxes, with a fraction of positives that is no larger than
        ``self.positive_fraction``.

        Args:
            See :meth:`ROIHeads.forward`

        Returns:
            list[Instances]:
                length `N` list of `Instances`s containing the proposals
                sampled for training. Each `Instances` has the following fields:

                - proposal_boxes: the proposal boxes
                - gt_boxes: the ground-truth box that the proposal is assigned to
                  (this is only meaningful if the proposal has a label > 0; if label = 0
                  then the ground-truth box is random)

                Other fields such as "gt_classes", "gt_masks", that's included in `targets`.
        """
        # Augment proposals with ground-truth boxes.
        # In the case of learned proposals (e.g., RPN), when training starts
        # the proposals will be low quality due to random initialization.
        # It's possible that none of these initial
        # proposals have high enough overlap with the gt objects to be used
        # as positive examples for the second stage components (box head,
        # cls head, mask head). Adding the gt boxes to the set of proposals
        # ensures that the second stage components will have some positive
        # examples from the start of training. For RPN, this augmentation improves
        # convergence and empirically improves box AP on COCO by about 0.5
        # points (under one tested configuration).
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(targets, proposals)

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )

            # Set target attributes of the sampled proposals:
            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes

            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                # We index all the attributes of targets that start with "gt_"
                # and have not been added to proposals yet (="gt_classes").
                # NOTE: here the indexing waste some compute, because heads
                # like masks, keypoints, etc, will filter the proposals again,
                # (by foreground/background, or number of keypoints in the image, etc)
                # so we essentially index the data twice.
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
            # If no GT is given in the image, we don't know what a dummy gt value can be.
            # Therefore the returned proposals won't have any gt_* fields, except for a
            # gt_classes full of background label.

            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        return proposals_with_gt


class MyMRCNN2(GeneralizedRCNN):
    def __init__(self, cfg):
        super().__init__(cfg)

    def forward(self, image_sizes, clip_feature, text_embeddings, gt_instances=None):
        if not self.training:
            return self.inference(image_sizes, clip_feature, text_embeddings)

        features = self.backbone(clip_feature)
        proposals, proposal_losses = self.proposal_generator(image_sizes, features, gt_instances)  # 这里会产生初始框proposal的损失
        _, detector_losses = self.roi_heads(features, proposals, text_embeddings, gt_instances)

        """可视化不在这里进行了
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)
        """

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    def inference(self, image_sizes, clip_feature, text_embeddings):
        features = self.backbone(clip_feature)
        proposals, _ = self.proposal_generator(image_sizes, features)

        results, scores = self.roi_heads(features, proposals, text_embeddings)

        return results, scores, features

    @classmethod
    def from_pretrained(cls, cfg, weight_path, device="cuda"):
        model = cls(cfg)
        checkpointer = DetectionCheckpointer(model)
        checkpointer.load(weight_path)

        # fpn 的 bottom_up 改为 CLIP 专属的 Adapter
        model.backbone.bottom_up = CLIPtoFPNAdapter()

        # 各类型改为更便利的子类
        model.proposal_generator.__class__ = MyRPN

        model.roi_heads.__class__ = MyROIHeads
        model.roi_heads.projection_adapter = nn.Linear(1024, 512)  # 还需要添加一个投影来对齐类别特征与文本embeddings

        model.roi_heads.box_predictor.__class__ = MyBoxPredictor
        model.roi_heads.box_predictor.cls_score = None  # 移除不需要的参数

        model.roi_heads.mask_head.__class__ = MyMaskHead

        return model


class CLIPtoFPNAdapter(nn.Module):
    def __init__(self, in_channels=768, out_channels_list=[2048, 1024, 512, 256]):
        super().__init__()
        self.convs = nn.ModuleList()
        for out_ch in out_channels_list:
            self.convs.append(nn.Conv2d(in_channels, out_ch, kernel_size=1))

        self.apply(init_weights_xavier)

    def forward(self, x):
        # x: [B, 768, 7, 7]
        c5 = self.convs[0](x)  # [B, 256, 7, 7]
        c4 = self.convs[1](F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False))  # [B, 256, 14, 14]
        c3 = self.convs[2](F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False))  # [B, 256, 28, 28]
        c2 = self.convs[3](F.interpolate(x, scale_factor=8, mode='bilinear', align_corners=False))  # [B, 256, 56, 56]

        features = {
            "res2": c2,
            "res3": c3,
            "res4": c4,
            "res5": c5,
        }
        return features

class MyDataset(Dataset):
    def __init__(self, dataset_name=None, dataset_dicts=None, meta_data=None, requires_instances=False, half=False):
        if dataset_name is None:
            if dataset_dicts is None or meta_data is None:
                raise ValueError("dataset_name为None时dataset_dicts与meta_data不能为None")
            else:
                self.dataset_dicts = dataset_dicts
                self.meta_data = meta_data
        else:
            self.dataset_dicts = DatasetCatalog.get(dataset_name)
            self.meta_data = MetadataCatalog.get(dataset_name)

        self.require_instances = requires_instances
        self.half = half

    def __len__(self):
        return len(self.dataset_dicts)

    def __getitem__(self, idx):
        d = self.dataset_dicts[idx].copy()
        # 读取 cv2 图像
        org_img = cv2.imread(d["file_name"])
        # d['cv2'] = org_img  # 不需要了
        # 这一部分来自defaults的__call__
        # 转 tensor
        # img = self.aug.get_transform(org_img).apply_image(org_img)
        d['image'] = torch.as_tensor(org_img.astype("float32").transpose(2, 0, 1))
        if self.half:
            d['image'] = d['image'].half()

        # 模型还希望在训练时能够有'Instances'，这里也加上
        if self.require_instances:
            height, width = org_img.shape[:2]
            instances = Instances((height, width))
            boxes = []
            classes = []
            masks = []
            for ann in d["annotations"]:
                x, y, w, h = ann["bbox"]
                boxes.append([x, y, x + w, y + h])
                classes.append(ann["category_id"])
                if "segmentation" in ann:
                    seg = ann["segmentation"]
                    if isinstance(seg, list):
                        # polygon
                        mask = np.zeros((height, width), dtype=np.uint8)
                        for poly in seg:
                            poly = np.array(poly).reshape((-1, 2))
                            cv2.fillPoly(mask, [poly.astype(np.int32)], 1)
                        masks.append(mask)
                    elif isinstance(seg, dict):
                        # RLE
                        rle = seg
                        if isinstance(rle['counts'], bytes):
                            rle['counts'] = rle['counts'].decode('utf-8')  # 如果是 bytes
                        mask = mask_utils.decode(rle)
                        masks.append(mask.astype(np.uint8))

            if self.half:
                instances.gt_boxes = Boxes(torch.tensor(boxes, dtype=torch.float32).half())
                instances.gt_classes = torch.tensor(classes, dtype=torch.int64).half()
                instances.gt_masks = BitMasks(torch.tensor(np.stack(masks, axis=0), dtype=torch.uint8).half())
            else:
                instances.gt_boxes = Boxes(torch.tensor(boxes, dtype=torch.float32))
                instances.gt_classes = torch.tensor(classes, dtype=torch.int64)
                instances.gt_masks = BitMasks(torch.tensor(np.stack(masks, axis=0), dtype=torch.uint8))
            d["instances"] = instances

        return d


def build_batch_loader(dataset_name=None, dataset_dicts=None, meta_data=None, batch_size=1, shuffle=False,
                       requires_instances=False, half=False):
    if dataset_name is None:
        if dataset_dicts is None or meta_data is None:
            raise ValueError("dataset_name为None时dataset_dicts与meta_data不能为None")
        else:
            dataset = MyDataset(dataset_dicts=dataset_dicts, meta_data=meta_data, requires_instances=requires_instances, half=half)
    else:
        dataset = MyDataset(dataset_name=dataset_name, requires_instances=requires_instances, half=half)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda batch: batch
    )


class MyFVLM(nn.Module):
    def __init__(self, clip_path, cfg, mask_rcnn_weight_path, device="cpu", top_k=100, nms_thresh=0.5, score_thresh=0.05):
        super().__init__()
        self.device = device

        self.clip_model = TwoStageCLIPModel.from_pretrained(clip_path).to(device)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_path)
        self.mask_rcnn_model = MyMRCNN2.from_pretrained(cfg, mask_rcnn_weight_path).to(device)

        self.test_topk_per_image = top_k
        self.test_nms_thresh = nms_thresh
        self.test_score_thresh = score_thresh

        self.froze_VLM()

    def froze_VLM(self):
        for param in self.clip_model.parameters():
            param.requires_grad = False

    def forward(self, batched_inputs, text_embeddings):
        if not self.training:
            return self.inference(batched_inputs, text_embeddings)

        batch_imgs = [bat["image"] for bat in batched_inputs]

        post_imgs = self.clip_processor(images=batch_imgs, return_tensors='pt', padding=True).to(
            self.device)  # TODO: 训练前做好是否可能？

        orig_size = post_imgs["pixel_values"].shape[-1]

        clip_feature = self.clip_model.get_image_features_stage1(post_imgs["pixel_values"])
        clip_feature = clip_feature.permute(0, 2, 1)
        clip_feature = clip_feature[:, :, 1:]
        last_dim_len = int(np.sqrt(clip_feature.shape[2]))
        clip_feature = clip_feature.reshape(clip_feature.shape[0], clip_feature.shape[1], last_dim_len, last_dim_len)

        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]

        image_sizes = [(orig_size, orig_size) for bat in batched_inputs]

        loss_dict = self.mask_rcnn_model(image_sizes, clip_feature, text_embeddings, gt_instances)

        return loss_dict

    def inference(self, batched_inputs, text_embeddings):
        batch_imgs = [bat["image"] for bat in batched_inputs]

        post_imgs = self.clip_processor(images=batch_imgs, return_tensors='pt', padding=True).to(
            self.device)  # TODO: 训练前做好是否可能？

        orig_size = post_imgs["pixel_values"].shape[-1]

        clip_feature = self.clip_model.get_image_features_stage1(post_imgs["pixel_values"])
        clip_feature = clip_feature.permute(0, 2, 1)
        clip_feature_cls = clip_feature[:, :, 0]
        clip_feature = clip_feature[:, :, 1:]
        last_dim_len = int(np.sqrt(clip_feature.shape[2]))
        clip_feature = clip_feature.reshape(clip_feature.shape[0], clip_feature.shape[1], last_dim_len, last_dim_len)

        image_sizes = [(orig_size, orig_size) for bat in batched_inputs]

        results, scores_1, backbone_features = self.mask_rcnn_model(image_sizes, clip_feature, text_embeddings)

        boxes_list = [proposal.pred_boxes for proposal in results]
        pooler_fmt_boxes = convert_boxes_to_pooler_format(boxes_list)
        del boxes_list

        # clip_feature可以作为Top-Level Feature Map直接进行ROI Align。原图尺寸也要取orig_size
        roi_align = ROIAlign(output_size=(last_dim_len, last_dim_len), spatial_scale=last_dim_len / orig_size,
                             sampling_ratio=2, aligned=True).to(self.device)
        clip_feature = roi_align(clip_feature, pooler_fmt_boxes)  # 获得逐个proposal的embedding
        clip_feature = clip_feature.reshape(clip_feature.shape[0], clip_feature.shape[1], -1)

        scores_2_list = []
        for i in range(0, clip_feature.shape[0], len(batched_inputs)):
            clip_feature_cls_slice = clip_feature_cls[pooler_fmt_boxes[i:min(i + len(batched_inputs), clip_feature.shape[0]), 0].long()].reshape(-1, clip_feature_cls.shape[-1], 1)
            clip_feature_slice = torch.cat((clip_feature_cls_slice, clip_feature[i:min(i + len(batched_inputs), clip_feature.shape[0])]), dim=2).permute(0, 2, 1)
            clip_feature_slice = self.clip_model.get_image_features_stage2(clip_feature_slice)
            scores_2_list.append(clip_feature_slice @ text_embeddings.T)

            torch.cuda.empty_cache()

        scores_2 = torch.cat(scores_2_list, dim=0)
        scores_2 = scores_2.softmax(dim=-1)
        scores = torch.sqrt(scores_1 * scores_2)
        scores = scores / scores.sum(dim=-1, keepdim=True)
        results = self.post_process_with_scores(results, scores, backbone_features, [(bat["image"].shape[-2], bat["image"].shape[-1]) for bat in batched_inputs])
        return results

    def post_process_with_scores(self, instances, scores, backbone_features, image_sizes):
        results, _ = fast_rcnn_inference(
            [p.pred_boxes.tensor for p in instances],
            scores.split([len(p.pred_boxes) for p in instances], dim=0),
            [p.image_size for p in instances],
            self.test_score_thresh,
            self.test_nms_thresh,
            self.test_topk_per_image
        )
        results = self.mask_rcnn_model.roi_heads._forward_mask(backbone_features, results)

        processed_results = []
        for results_per_image, img_size in zip(results, image_sizes):
            r = detector_postprocess(results_per_image, img_size[0], img_size[1])
            processed_results.append({"instances": r})

        return processed_results

    def class_name_list_prepare(self, class_name_list):
        class_name_list = ["a photo of " + cls for cls in class_name_list]
        class_name_list.append("a random photo with no specific object")
        return class_name_list

    def get_cls_embedding(self, class_name_list):
        class_inputs = self.clip_processor(text=class_name_list, return_tensors="pt", padding=True).to(
            self.clip_model.device)
        return self.clip_model.get_text_features(**class_inputs).to(self.clip_model.device)

    def save(self, path):
        torch.save({"model": self.mask_rcnn_model.state_dict()}, path)

    def load(self, path, device):
        self.device = device
        checkpoint = torch.load(path, map_location=device)
        self.mask_rcnn_model.load_state_dict(checkpoint["model"])


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@torch.no_grad()
def test(model, batch_size, dataset_name, dataset_path, dataset_json, model_load_path=None, evaluator_output_path=None,
         shuffle=False, half=False, visualize_=False, visualize_path=None):
    if dataset_name not in DatasetCatalog.list():
        register_coco_instances(dataset_name, {}, dataset_json, dataset_path)
    dataset_dicts = DatasetCatalog.get(dataset_name)
    meta_data = MetadataCatalog.get(dataset_name)

    loader = build_batch_loader(dataset_dicts=dataset_dicts, meta_data=meta_data,
                                batch_size=batch_size, shuffle=shuffle, requires_instances=True, half=half)

    autocast_ctx = autocast(dtype=torch.float16) if half else nullcontext()

    evaluator = COCOEvaluator(dataset_name if dataset_name else "custom_coco", output_dir=evaluator_output_path)
    evaluator.reset()

    if model_load_path:
        model.load(model_load_path, model.device)

    model.eval()

    with torch.no_grad(), autocast_ctx:
        class_name_list = model.class_name_list_prepare(meta_data.thing_classes)
        class_embeddings = model.get_cls_embedding(class_name_list)


    for batch in loader:
        results = []  # 不要一次性存太多results了
        with torch.no_grad(), autocast_ctx:
            pred_instances = model.inference(batch, class_embeddings)

            # 转换为 Detectron2 标准格式
            for det, inp in zip(pred_instances, batch):
                out_dict = {
                    "image_id": inp["image_id"],
                    "instances": det["instances"].to("cpu"),
                    "height": inp["height"],
                    "width": inp["width"],
                    "image": inp["image"].cpu(),
                    "file_name": inp["file_name"]
                }
                evaluator.process([inp], [out_dict])
                results.append(out_dict)
            del pred_instances, batch
            torch.cuda.empty_cache()

            if visualize_:
                visualize(results, visualize_path, meta_data)

    metrics = evaluator.evaluate()
    return metrics


def visualize(results, visualize_path, meta_data):
    for result in results:
        img = result["image"].permute(1, 2, 0).numpy()
        img = img[:, :, ::-1]

        # 创建 Visualizer
        v = Visualizer(img, metadata=meta_data, scale=1.2)

        out = v.draw_instance_predictions(result["instances"])
        img_vis = out.get_image()

        img_path = os.path.join(visualize_path, f"{os.path.basename(result['file_name'])}")
        if not cv2.imwrite(img_path, img_vis[..., ::-1]):  # 转回 BGR
            raise IOError(f"Failed to visualize image {img_path}")
        print(f"saved at {img_path}")


def save_training_state(model, optimizer, scheduler, scaler, epoch, path):
    torch.save({
        "epoch": epoch,
        "model": model.mask_rcnn_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "scaler": scaler.state_dict() if scaler else None,
    }, path)


def load_training_state(model, optimizer, scheduler, scaler, path, device="cuda"):
    checkpoint = torch.load(path, map_location=device)
    if checkpoint.get("model") and model.mask_rcnn_model:
        model.mask_rcnn_model.load_state_dict(checkpoint["model"])
    if checkpoint.get("optimizer") and optimizer:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if checkpoint.get("scheduler") and scheduler:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if checkpoint.get("scaler") and scaler:
        scheduler.load_state_dict(checkpoint["scaler"])
    epoch = checkpoint.get("epoch", 0)
    return epoch


def train(model, optimizer, scheduler, scaler, epoch_num, batch_size, dataset_name, dataset_path, dataset_json, start_epoch=0, shuffle=False,
          save_epoch=0, train_name="test", half=False):
    if dataset_name not in DatasetCatalog.list():
        register_coco_instances(dataset_name, {}, dataset_json, dataset_path)
    dataset_dicts = DatasetCatalog.get(dataset_name)
    meta_data = MetadataCatalog.get(dataset_name)

    loader = build_batch_loader(dataset_dicts=dataset_dicts, meta_data=meta_data,
                                batch_size=batch_size, shuffle=shuffle, requires_instances=True, half=half)

    autocast_ctx = autocast(dtype=torch.float16) if half else nullcontext()

    with torch.no_grad(), autocast_ctx:
        class_name_list = model.class_name_list_prepare(meta_data.thing_classes)
        class_embeddings = model.get_cls_embedding(class_name_list)

    all_epoch_losses = []  # 保存每个epoch的平均loss

    model.train()

    for epoch in range(start_epoch, epoch_num):
        epoch_loss_sum = {}
        cnt = 0
        for batch in loader:
            optimizer.zero_grad()

            if cnt == 0 and epoch == 0:
                for name, param in model.mask_rcnn_model.named_parameters():
                    if not ("projection_adapter" in name or "bottom_up" in name or "roi_heads.box_predictor.bbox_pred" in name):
                        param.requires_grad = False

            if cnt == 400 and epoch == 0:
                for name, param in model.mask_rcnn_model.named_parameters():
                    if not ("projection_adapter" in name or "bottom_up" in name or "roi_heads.box_predictor.bbox_pred" in name):
                        param.requires_grad = True

            with autocast_ctx:
                losses = model(batch, class_embeddings)
                total_loss = sum(losses.values())
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()  # 每个batch调用一次
                for k, v in losses.items():
                    epoch_loss_sum[k] = epoch_loss_sum.get(k, 0.0) + v.item()

                cnt += 1
                if cnt % 20 == 0:
                    # 获取当前学习率（假设optimizer只有一个参数组）
                    current_lr = optimizer.param_groups[0]['lr']
                    print(f"Epoch {epoch}/{epoch_num}, Batch {cnt}/{len(loader)}, LR {current_lr}")

                    epoch_loss_avg = {k: v / len(loader) for k, v in epoch_loss_sum.items()}
                    all_epoch_losses.append(epoch_loss_avg)

                    # 计算总平均 loss
                    total_epoch_loss = sum(epoch_loss_avg.values())

                    # 打印信息
                    for k, v in epoch_loss_avg.items():
                        print(f"  {k}: {v:.6f}")
                    print(f"  Total Loss: {total_epoch_loss:.6f}\n")

                    epoch_loss_sum = {}

        if epoch % save_epoch == 0:
            save_training_state(model, optimizer, scheduler, scaler, epoch,
                                f"../model/checkpoint/{train_name}_{dataset_name}_{epoch}.pth")
            print(f"Saved checkpoint for epoch {epoch + 1}")


def test_one_image(model, image_path, class_name_list, model_load_path=None, half=False, visualize_=False, visualize_path=None):
    autocast_ctx = autocast(dtype=torch.float16) if half else nullcontext()
    if model_load_path:
        model.load(model_load_path, model.device)

    with torch.no_grad(), autocast_ctx:
        long_class_name_list = model.class_name_list_prepare(class_name_list)
        class_embeddings = model.get_cls_embedding(long_class_name_list)

    d = {}
    org_img = cv2.imread(image_path)
    d['image'] = torch.as_tensor(org_img.astype("float32").transpose(2, 0, 1))
    if half:
        d['image'] = d['image'].half()
    d['height'] = org_img.shape[0]
    d['width'] = org_img.shape[1]
    d['file_name'] = image_path

    model.eval()

    with torch.no_grad(), autocast_ctx:
        pred_instances = model.inference([d], class_embeddings)
        out_dict = {
            "instances": pred_instances[0]["instances"].to("cpu"),
            "height": d["height"],
            "width": d["width"],
            "image": d["image"].cpu(),
            "file_name": d["file_name"]
        }

    if visualize_:
        visualize([out_dict], visualize_path, {"thing_classes": class_name_list})
