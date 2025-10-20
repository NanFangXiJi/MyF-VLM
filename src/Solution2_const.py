from Solution2 import get_cfg, model_zoo


#val_json = "../data/COCO/annotations/instances_val2017.json"
filtered_val_json = "../data/COCO/annotations/instances_val2017_filtered.json"
val_images = "../data/COCO/val2017"

#train_json = "../data/COCO/annotations/instances_train2017.json"
filtered_train_json = "../data/COCO/annotations/instances_train2017_filtered.json"
train_images = "../data/COCO/train2017"

val_small_json = "../data/COCO/annotations/val_small.json"

MRCNN_PATH = "../model/model_final_f10217.pkl"
CONFIG_FILE = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"

CLIP_PATH = "../model/clip-vit-patch32/models--openai--clip-vit-base-patch32/snapshots/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
MyMRCNN_PATH = "../model/my_mask_rcnn.pkl"
MyMRCNN_CLSFREE_PATH = "../model/my_clsfree_mask_rcnn.pkl"

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(CONFIG_FILE))
cfg.MODEL.WEIGHTS = MRCNN_PATH
cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG = True
cfg.MODEL.DEVICE = "cuda"

deterministic = False
half = True

my_trained_model = "../model/FVLM_final.pth"

