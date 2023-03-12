from ultralytics import YOLO
from PIL import Image
import cv2
from pathlib import Path

model = YOLO("../../yolov8/runs/detect/train15/weights/best.pt")

image_files_list = []


# from ndarray
# im2 = cv2.imread("bus.jpg")
# print(im2)
results = model.predict(source="../../datasets/from-roboflow/fscbp-proj2/train/images/161678450999_-pic_jpg.rf.f6ca9f4f129050ea891747698b9c33d0.jpg", save=True, save_txt=True)  # save predictions as labels
# results = model.predict(source="coco128.yaml", save=True, save_txt=True)  # save predictions as labels

print(results)