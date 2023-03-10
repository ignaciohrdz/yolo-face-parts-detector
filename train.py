from ultralytics import YOLO
import os
from utils import *


if __name__ == "__main__":

    path_dataset = os.path.join(PATH_HOME, "Documents", "Datasets", "Face-Parts-Dataset")
    path_yaml = os.path.join(path_dataset, "split", "data.yaml")

    # Creating the model
    model = YOLO("weights/yolov8n.pt")

    # Train the model
    results = model.train(data=path_yaml, task="detect", name="train",
                          epochs=10, workers=4, batch=8,
                          scale=0.25, degrees=25.0, mosaic=0.8)

