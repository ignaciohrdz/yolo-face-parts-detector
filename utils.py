from pathlib import Path
import cv2
import platform
import getpass
import numpy as np


def smart_resize(img, new_size=512):
    ratio = new_size/max(img.shape[:2])
    return cv2.resize(img, None, fx=ratio, fy=ratio), ratio


def points_to_YOLO(labels_df, points, part_id, img_h, img_w):
    # Create a contour from the list of X,Y points and get the bounding box
    contour = np.array(points, dtype=np.int32)
    x, y, w, h = cv2.boundingRect(contour)
    x_n, w_n = x / img_w, w / img_w
    y_n, h_n = y / img_h, h / img_h
    x_c = x_n + 0.5 * w_n
    y_c = y_n + 0.5 * h_n

    # Populating the dataframe
    labels_df.loc[len(labels_df), :] = [part_id, x_c, y_c, w_n, h_n]
    return x, y, w, h  # these are not the normalised coordinates, these are for plotting the box


PATH_HOME = Path.home() if not platform.system() == "Linux" else "/mnt/c/Users/{}".format(getpass.getuser())
