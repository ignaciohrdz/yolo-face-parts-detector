import os
import argparse

from ultralytics import YOLO
import cv2
import imageio
import supervision as spv


if __name__ == "__main__":

    default_model_path = os.path.join("runs", "detect", "train_n")

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", '--model_dir', type=str, default=default_model_path, help="Path to the model weights")
    parser.add_argument('--save_gif', action="store_true", help="Save the video to a GIF file")
    args = parser.parse_args()

    # Loading the model
    try:
        path_model = args.model_dir
        model = YOLO(os.path.join(path_model, "weights", "best.pt"))
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
        path_gif = os.path.join(path_model, "live_demo.gif")

    while True:
        ret, frame = cap.read()

        result = model(frame, agnostic_nms=True)[0]
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

        if k == ord("q"):
            break

    cv2.destroyAllWindows()
    cap.release()

    # Exporting to GIF
    # Source: https://pysource.com/2021/03/25/create-an-animated-gif-in-real-time-with-opencv-and-python/
    if args.save_gif:
        print("\nSaving the stream to ", path_gif)
        with imageio.get_writer(path_gif, mode="I") as writer:
            for f in frames:
                writer.append_data(f)
