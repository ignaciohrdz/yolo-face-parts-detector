from ultralytics import YOLO
import os
import argparse
from utils import *


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-a", '--arch', type=str, default='n', help="Architecture (n, s, m, l, x)")
    parser.add_argument("-d", '--data_dir', type=str, help="Path to the datasets folder")
    parser.add_argument("-b", '--batch_size', type=int, default=8, help="Batch size")
    parser.add_argument("-e", '--epochs', type=int, default=10, help="Number of epochs")
    parser.add_argument("--device", type=str, default=['0'], nargs='+', help="Device list (also accepts cpu)")
    args = parser.parse_args()

    if args.data_dir is not None:
        path_datasets = args.data_dir
    else:
        path_datasets = os.path.join(PATH_HOME, "Documents", "Datasets")

    path_face_parts = os.path.join(path_datasets, "Face-Parts-Dataset")
    path_yaml = os.path.join(path_face_parts, "split", "data.yaml")

    # Training all models
    model = YOLO("weights/yolov8{}.pt".format(args.arch))

    # Train the model
    results = model.train(data=path_yaml, task="detect", name="train_{}".format(args.arch),
                          epochs=args.epochs, batch=args.batch_size,
                          device=",".join(args.device),
                          scale=0.25, degrees=25.0, mosaic=0.8)
