from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json


json_gt = "../data/COCO/annotations/instances_val2017_filtered.json"
json_pred = "../output/output_final/eval/coco_instances_results.json"

# 加一些 COCO 非得要的东西
with open(json_gt, "r", encoding="utf-8") as f:
    data = json.load(f)

if "info" not in data:
    data["info"] = {"description": "auto-filled COCO info"}
if "licenses" not in data:
    data["licenses"] = []


temp_gt = json_gt.replace(".json", "_fixed.json")
with open(temp_gt, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False)


coco_gt = COCO(temp_gt)
coco_dt = coco_gt.loadRes(json_pred)


for iou_type in ["bbox", "segm"]:
    print(f"\n🔹 Evaluating {iou_type.upper()} ...")
    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


    metrics = {
        "AP": coco_eval.stats[0],
        "AP50": coco_eval.stats[1],
        "AP75": coco_eval.stats[2],
        "APs": coco_eval.stats[3],
        "APm": coco_eval.stats[4],
        "APl": coco_eval.stats[5],
        "AR@1": coco_eval.stats[6],
        "AR@10": coco_eval.stats[7],
        "AR@100": coco_eval.stats[8],
        "AR_l": coco_eval.stats[9],
    }
    for k, v in metrics.items():
        print(f"{k:<6}: {v:.4f}")
