from ultralytics import YOLO
import os
import cv2
import supervision as spv


if __name__ == "__main__":

    path_model = os.path.join("runs", "detect", "train", "weights")

    # Creating the model
    model = YOLO(os.path.join(path_model, "best.pt"))

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

        if k == ord("q"):
            break
