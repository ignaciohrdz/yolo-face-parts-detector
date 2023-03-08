from ultralytics import YOLO
import os

if __name__ == "__main__":

    path_model = os.path.join("runs", "detect", "train", "weights")

    # Creating the model
    model = YOLO(os.path.join(path_model, "best.pt"))
    results = model.predict(source="0", show=True)

