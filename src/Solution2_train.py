from Solution2 import MyFVLM, init_weights_xavier, cosine_scheduler_with_warmup, GradScaler, train, load_training_state
from Solution2_const import *
import torch


model = MyFVLM(CLIP_PATH, cfg, MRCNN_PATH, "cuda", score_thresh=0.2, nms_thresh=0.3)

adapter_params = []
backbone_params = []

for name, param in model.mask_rcnn_model.named_parameters():
    if "projection_adapter" in name or "bottom_up" in name or "roi_heads.box_predictor.bbox_pred" in name or "mask" in name:
        adapter_params.append(param)
        module_name = ".".join(name.split(".")[:-1])
        if module_name:
            module = model.mask_rcnn_model.get_submodule(module_name)
            init_weights_xavier(module)
    else:
        backbone_params.append(param)

optimizer = torch.optim.AdamW([
    {"params": backbone_params, "lr": 1e-5},
    {"params": adapter_params, "lr": 1e-4}
], weight_decay=1e-4)
scheduler = cosine_scheduler_with_warmup(optimizer, warmup_steps=400, total_steps=22000)
scaler = GradScaler(enabled=half)
start_epoch = 0
start_epoch = load_training_state(model, optimizer, scheduler, scaler, my_trained_model, model.device) + 1  # 如果从头训练，注释此行

train(model, optimizer, scheduler, scaler, 5, 15, "complete_train",
      train_images, filtered_train_json, start_epoch, False, 1, "AllOrNothing", half)