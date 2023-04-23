from ultralytics import YOLO
import os
import argparse
from utils import *


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", '--data_dir', type=str, help="Path to the datasets folder")
    parser.add_argument("-b", '--batch_size', type=int, default=8, help="Batch size")
    parser.add_argument("-w", '--num_workers', type=int, default=0, help="Number of workers")
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
    for m in ['n', 's', 'm', 'l', 'x']:
        model = YOLO("weights/yolov8{}.pt".format(m))

        # Train the model
        results = model.train(data=path_yaml, task="detect", name="train_{}".format(m),
                              epochs=args.epochs, workers=args.num_workers, batch=args.batch_size,
                              device=",".join(args.device),
                              scale=0.25, degrees=25.0, mosaic=0.8)
