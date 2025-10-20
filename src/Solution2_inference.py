from Solution2 import MyFVLM, test, set_seed
from Solution2_const import *
import torch


if deterministic:
    set_seed(0)
else:
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


model = MyFVLM(CLIP_PATH, cfg, MyMRCNN_CLSFREE_PATH, "cuda", score_thresh=0.2, nms_thresh=0.3)

test(model, 50, "val_test", val_images, filtered_val_json, model_load_path=my_trained_model,
     evaluator_output_path="../output/output_final/eval/", shuffle=False, half=half, visualize_=True, visualize_path="../output/output_final/img/")