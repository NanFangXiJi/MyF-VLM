import json
import numpy as np
from pycocotools.coco import COCO
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)


gt_path = "../data/COCO/annotations/instances_val2017_filtered.json"
pred_path = "../output/output_final/eval/coco_instances_results.json"
temp_gt = gt_path.replace(".json", "_fixed.json")



coco_gt = COCO(temp_gt)
with open(pred_path, "r") as f:
    coco_pred = json.load(f)


gt_classes = []
pred_classes = []

for ann in coco_gt.dataset["annotations"]:
    img_id = ann["image_id"]
    gt_cls = ann["category_id"]

    # 获取对应预测结果（取该图片中得分最高的预测类别）
    preds = [p for p in coco_pred if p["image_id"] == img_id]
    if len(preds) == 0:
        continue
    best_pred = max(preds, key=lambda x: x["score"])
    pred_cls = best_pred["category_id"]

    gt_classes.append(gt_cls)
    pred_classes.append(pred_cls)

gt_classes = np.array(gt_classes)
pred_classes = np.array(pred_classes)


acc = accuracy_score(gt_classes, pred_classes)
precision = precision_score(gt_classes, pred_classes, average='macro', zero_division=0)
recall = recall_score(gt_classes, pred_classes, average='macro', zero_division=0)
f1 = f1_score(gt_classes, pred_classes, average='macro', zero_division=0)
cm = confusion_matrix(gt_classes, pred_classes)


print("===== Classification Evaluation =====")
print(f"Accuracy  : {acc:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-score  : {f1:.4f}")
print("\nPer-class metrics:")
print(classification_report(gt_classes, pred_classes, target_names=[c['name'] for c in coco_gt.loadCats(coco_gt.getCatIds())], digits=4))
print("\nConfusion Matrix:")
print(cm)
