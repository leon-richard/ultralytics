from ultralytics import YOLO

model = YOLO()
model.train(cfg="default_copy.yaml")