import re
import numpy as np
from pathlib import Path
import shutil

from ultralytics import YOLO
from ultralytics.yolo.utils import yaml_load
from ultralytics.yolo.data.utils import IMG_FORMATS

# 因为coco数据集的80个labels包括了bike、person、motobike等，我们也需要用这些类别，
# 于是把80个labels的最后2个label替换为smoke和fire，应该是我们的最佳选择。

# Constants
LABELS_MAPPING_PATH = Path.cwd() / 'labels-mapping'
MERGE_CFG_PATH = LABELS_MAPPING_PATH / 'merge-cfg.yaml'
MERGE_CFG_DICT = yaml_load(MERGE_CFG_PATH)

IMAGE_FILES_PATH = Path.cwd() / MERGE_CFG_DICT["image-files-path"]
MANUAL_LABEL_FILES_PATH = Path.cwd() / MERGE_CFG_DICT["manual-label-files-path"]
YOLOV8X_LABEL_FILES_PATH = LABELS_MAPPING_PATH / MERGE_CFG_DICT["yolov8x-label-files-path"]

if not IMAGE_FILES_PATH.exists():
    raise NotADirectoryError(f"image files path {IMAGE_FILES_PATH} incorrect!")

if not MANUAL_LABEL_FILES_PATH.exists():
    raise NotADirectoryError(f"manual label files path {MANUAL_LABEL_FILES_PATH} incorrect!")

if YOLOV8X_LABEL_FILES_PATH.exists():
    try:
        shutil.rmtree(YOLOV8X_LABEL_FILES_PATH)
    except OSError as e:
        print("Error: %s : %s" % (YOLOV8X_LABEL_FILES_PATH, e.strerror))
    # YOLOV8X_LABEL_FILES_PATH.rmdir()
YOLOV8X_LABEL_FILES_PATH.mkdir(parents=True)

# 解析manual-label-names到merged-label-names的关系，形成mapping表
manual_label_mapping = {}
for from_key, from_val in MERGE_CFG_DICT['manual-label-names'].items():
    to_key = -1
    for k, to_val in MERGE_CFG_DICT['merged-label-names'].items():
        if from_val == to_val:
            to_key = k
            break
    manual_label_mapping[from_key] = to_key
print(manual_label_mapping)

# 解析yolov8-label-names到merged-label-names的关系，形成mapping表
yolov8_label_mapping = {}
for from_key, from_val in MERGE_CFG_DICT['yolov8-label-names'].items():
    to_key = -1
    for k, to_val in MERGE_CFG_DICT['merged-label-names'].items():
        if from_val == to_val:
            to_key = k
            break
    yolov8_label_mapping[from_key] = to_key
print(yolov8_label_mapping)

model = YOLO("yolov8x.pt")

# 步骤：
# 1. 用yolov8x自动为images生成labels，并通过mapping删除78和79号标签；
for image_file_path in IMAGE_FILES_PATH.rglob("*.jpg"):
    if image_file_path.is_file():
        results = model.predict(source=image_file_path, save=True, save_txt=True)
        
        save_label_path = str(YOLOV8X_LABEL_FILES_PATH / image_file_path.stem)

        class_id, xywhn = results[0].boxes.cls.view(-1,1).cpu().numpy(), results[0].boxes.xywhn.cpu().numpy()
        numpy_boxes = np.concatenate((class_id, xywhn), axis=1)

        # 把class_id进行mapping
        for row in numpy_boxes:                         # 遍历二维数组中的每一行
            row[0] = yolov8_label_mapping[row[0]]
        numpy_boxes = np.delete(numpy_boxes, np.where(numpy_boxes[:,0] == -1), axis=0)  # 删除映射成-1的行
        
        with open(f'{save_label_path}.txt', 'a') as f:
            np.savetxt(f, numpy_boxes, fmt='%g')
            # f.write(str(numpy_boxes))

# 2. 把别人手工标注好的labels，通过mapping映射到78和79号标签；
# 3. 按照文件名，把上述两类标签合并，并且把image文件也整理到输出目录。

