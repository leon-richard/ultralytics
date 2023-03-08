from ultralytics import YOLO
from PIL import Image
import cv2
from pathlib import Path

model = YOLO("yolov8x.pt")

image_files_list = []


# from ndarray
# im2 = cv2.imread("bus.jpg")
# print(im2)
results = model.predict(source="../../datasets/motofire/", save=True, save_txt=True)  # save predictions as labels
# results = model.predict(source="coco128.yaml", save=True, save_txt=True)  # save predictions as labels

# print(results)