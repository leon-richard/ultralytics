from ultralytics import YOLO
from PIL import Image
import cv2
from pathlib import Path

model = YOLO("../../yolov8/runs/detect/train15/weights/best.pt")

image_files_list = []


# from ndarray
# im2 = cv2.imread("bus.jpg")
# print(im2)
results = model.predict(source="../../datasets/from-roboflow/fscbp-proj2/valid/images/191678450999_-pic_jpg.rf.e8555dfbffbbe5fbd0cfbd0566b08fdc.jpg", save=True, save_txt=True)  # save predictions as labels
# results = model.predict(source="coco128.yaml", save=True, save_txt=True)  # save predictions as labels

print(results)