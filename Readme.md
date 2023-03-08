# Face parts detection with YOLOv8 👃

## Introduction

In this project I will use the most recent implementation of YOLO by Ultralytics, [YOLOv8](https://github.com/ultralytics/ultralytics). The goal is to train an algorithm that is able to detect separate face parts without having to use landmark detectors that don't do well when part of the face is occluded or missing. It is also a great opportunity to try out the `ultralytics` library.

## Data

For this experiment I'm using the following sources:

- Existing datasets: all these datasets were processed by converting each group of facial landmarks (eye, mouth, nose, eyebrows) to a bounding box compatible with YOLO.
  - [Helen dataset](http://www.ifp.illinois.edu/~vuongle2/helen/)
  - [Menpo2D dataset](https://github.com/jiankangdeng/MenpoBenchmark)
  - [AFW (Annotated Faces in the Wild)](https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/)
- Custom datasets:
  - [Pexels](https://pexels.com): I downloaded 85 images from this website and annotated them using [CVAT](https://app.cvat.ai/).