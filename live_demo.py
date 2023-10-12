from pathlib import Path
import argparse
import time

from ultralytics import YOLO
import cv2
import numpy as np
import imageio
import supervision as spv


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", '--path_model', type=str, help="Path to the model")
    parser.add_argument('--save_gif', action="store_true", help="Save the video to a GIF file")
    args = parser.parse_args()

    # Debugging
    # args.path_model = "runs/detect/train_n"
    # args.save_gif = True

    # Loading the model
    try:
        path_model = Path(args.path_model)
        model = YOLO(path_model / "weights" / "best.pt")
    except FileNotFoundError:
        print("ERROR: Could not load the YOLO model")
        exit()

    # This will draw the detections
    class_colors = spv.ColorPalette.from_hex(['#ffff66', '#66ffcc', '#ff99ff', '#ffcc99'])
    box_annotator = spv.BoxAnnotator(
        thickness=2,
        text_thickness=1,
        text_scale=0.5,
        color=class_colors
    )

    # Reading frames from the webcam
    cap = cv2.VideoCapture(0)

    # Optional: exporting to GIF
    if args.save_gif:
        frames = []
        times = []
        path_gif = path_model / "live_demo.gif"

    while True:
        ret, frame = cap.read()

        start_time = time.time()
        result = model(frame, agnostic_nms=True, verbose=False)[0]
        detections = spv.Detections.from_yolov8(result)

        labels = [
            f"{model.model.names[class_id]} {confidence:0.2f}"
            for _, confidence, class_id, _
            in detections
        ]
        frame = box_annotator.annotate(
            scene=frame,
            detections=detections,
            labels=labels
        )

        cv2.imshow("Face parts", frame)
        k = cv2.waitKey(1)

        if args.save_gif:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            times.append(time.time() - start_time)

        if k == ord("q"):
            break

    cv2.destroyAllWindows()
    cap.release()

    # Exporting to GIF
    # Source: https://pysource.com/2021/03/25/create-an-animated-gif-in-real-time-with-opencv-and-python/
    if args.save_gif:
        print("\nSaving the stream to ", path_gif)
        avg_time = np.array(times).mean()
        fps = round(1 / avg_time)
        imageio.mimsave(path_gif, frames, format='GIF', fps=fps)
