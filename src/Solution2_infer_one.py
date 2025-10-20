from Solution2 import  MyFVLM, test_one_image
from Solution2_const import *


image_path = "../data/my_image/my_dorm.jpg"
class_name_list = ["box", "lamp", "keyboard", "mouse", "laptop", "headset", "bag", "book", "chair", "person", "clothing", "umbrella", "coat hanger"]

model = MyFVLM(CLIP_PATH, cfg, MyMRCNN_CLSFREE_PATH, "cuda", score_thresh=0.2, nms_thresh=0.3)

test_one_image(model, image_path, class_name_list, my_trained_model, half, True, "../output/my_image_test")
