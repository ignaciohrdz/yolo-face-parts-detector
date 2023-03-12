"""
This script converts the annotations of several facial landmark detection datasets to YOLO format
It also adds some images from Pexels that I annotated on CVAT
The output of this code is a folder with everything needed to train a YOLOv8 model

Author: Ignacio Hernández Montilla, 2023
"""

from utils import *
import shutil
import glob

import pandas as pd
import os
import yaml

import random
random.seed(420)


def process_names(names, split, path_data, path_dest, skip):
    """
    This function reads the split data of all datasets
     and saves it following the YOLO folder structure
    :param names: Dataframe with a single column (the
    :param split: split name ("val" or "train")
    :param path_data: path to the data
    :param path_dest: path to the exported data
    :param skip: list of image names that we may want to skip
    :return: None
    """
    names[1] = names[0].apply(lambda x: os.path.join(path_data, "images", x + ".jpg"))
    names[2] = names[0].apply(lambda x: os.path.join(path_data, "labels", x + ".txt"))
    path_imgs_txt = os.path.join(path_dest, "images", split, "images.txt")
    path_labels_txt = os.path.join(path_dest, "labels", split, "labels.txt")
    use_imgs = names.loc[~names[0].isin(skip), 1]
    use_labels = names.loc[~names[0].isin(skip), 2]
    use_imgs.to_csv(path_imgs_txt, header=False, index=False)
    use_labels.to_csv(path_labels_txt, header=False, index=False)


if __name__ == "__main__":

    IMSHOW_WAIT_TIME = 33  # for cv2.imshow

    # Original data from Helen
    path_helen_dataset = os.path.join(PATH_HOME, "Documents", "Datasets", "Helen-dataset")
    path_helen_images = os.path.join(path_helen_dataset, "images")
    path_helen_annotations = os.path.join(path_helen_dataset, "annotation")

    # Original data from Pexels
    path_pexels_dataset = os.path.join(PATH_HOME, "Documents", "Datasets", "Pexels-face-parts")

    # Original data from the AFW dataset
    path_afw_dataset = os.path.join(PATH_HOME, "Documents", "Datasets", "AFW-dataset")

    # Original data from the Menpo2D dataset
    path_menpo2D_dataset = os.path.join(PATH_HOME, "Documents", "Datasets", "Menpo2D")

    # The results will go here
    path_processed_dataset = Path(os.path.join(PATH_HOME, "Documents", "Datasets", "Face-Parts-Dataset"))
    path_processed_images = Path(os.path.join(path_processed_dataset, "images"))
    path_processed_labels = Path(os.path.join(path_processed_dataset, "labels"))
    path_yolo_data = Path(os.path.join(path_processed_dataset, "split"))
    path_yolo_images = Path(os.path.join(path_yolo_data, "images"))
    path_yolo_labels = Path(os.path.join(path_yolo_data, "labels"))

    # Create the YOLO folders if they don't exist
    path_yolo_data.mkdir(parents=True, exist_ok=True)
    for s in ["train", "val"]:
        Path(path_yolo_images/s).mkdir(parents=True, exist_ok=True)
        Path(path_yolo_labels/s).mkdir(parents=True, exist_ok=True)
    path_processed_dataset.mkdir(parents=True, exist_ok=True)
    path_processed_images.mkdir(parents=True, exist_ok=True)
    path_processed_labels.mkdir(parents=True, exist_ok=True)

    ########################################
    # PART 1: PROCESSING THE HELEN DATASET #
    ########################################

    """
    How should I deal with the images of the Helen dataset that have more than one face?
    It's too complicated to deal with these images because we would have to
    stitch each group of images into a single one (plus correcting the coordinates)
    That's why I will simply exclude them from the processed dataset
    """
    img_face_df = pd.DataFrame({'img_name': sorted(os.listdir(path_helen_images))})
    img_face_df['img_id'] = img_face_df['img_name'].str.split("_").str[0]
    img_face_df['face_id'] = img_face_df['img_name'].str.split("_").str[1].str.replace(".jpg", "").astype(int)
    face_counts = img_face_df[['img_id', 'face_id']].groupby(['img_id']).size().to_frame(name='face_count').reset_index()

    imgs_multi_faces = face_counts.loc[face_counts.face_count > 1, 'img_id']
    skip_imgs_df = img_face_df.loc[(img_face_df['img_name'].str.split("_").str[0].isin(imgs_multi_faces)) |
                                   (img_face_df.face_id > 1), :]
    skip_imgs = skip_imgs_df.img_name.to_list()
    print("There are {} samples with more than one face".format(len(skip_imgs)))

    # Linking each face part with its corresponding points
    part_points_helen = {'jaw': [list(range(0, 41))],
                         'eye': [list(range(114, 134)), list(range(134, 154))],  # left and right
                         'nose': [list(range(41, 58)) + [154, 174]],
                         'mouth': [list(range(58, 114))],
                         'eyebrow': [list(range(154, 174)), list(range(174, 194))]}  # left and right
    use_parts = [p for p in part_points_helen.keys() if p != "jaw"]

    for ann in os.listdir(path_helen_annotations):
        with open(os.path.join(path_helen_annotations, ann)) as f:
            lines = [l.rstrip() for l in f.readlines()]

        img_name = lines.pop(0)
        img_name = "{}.jpg".format(img_name)

        if img_name not in skip_imgs:
            print("{}: {} landmarks".format(img_name, len(lines)))

            img = cv2.imread(os.path.join(path_helen_images, img_name), cv2.IMREAD_COLOR)
            img_h, img_w = img.shape[:2]
            img, ratio = smart_resize(img)

            cv2.imshow("Image", img)
            cv2.waitKey(IMSHOW_WAIT_TIME)

            # Converting each face part into a bounding box (using the YOLO format)
            img_labels = pd.DataFrame(columns=['class', 'x', 'y', 'w', 'h'])
            for part_id, part_name in enumerate(use_parts):
                for idxs in part_points_helen[part_name]:
                    points = [lines[i] for i in idxs]
                    contour = []
                    for i, p in enumerate(points):
                        x, y = [float(c) for c in p.split(" , ")]
                        contour.append([x, y])
                        img = cv2.circle(img, (int(x*ratio), int(y*ratio)), 3, (0, 255, 255), -1)

                    # Getting the bounding box in YOLO format
                    x, y, w, h = points_to_YOLO(img_labels, contour, part_id, img_h, img_w)

                    img = cv2.rectangle(img,
                                        (int(x*ratio), int(y*ratio)),
                                        (int((x+w)*ratio), int((y+h)*ratio)), (0, 0, 255), 2)

            cv2.imshow("Image", img)
            cv2.waitKey(IMSHOW_WAIT_TIME)

            # Saving the data
            img_source = os.path.join(path_helen_images, img_name)
            img_dest = os.path.join(path_processed_images, img_name)
            shutil.copy(img_source, img_dest)
            label_dest = os.path.join(path_processed_labels, os.path.splitext(img_name)[0] + ".txt")
            img_labels.round(6).to_csv(label_dest, header=False, index=False, sep=" ")

    ########################################
    # PART 2: PROCESSING THE PEXELS IMAGES #
    ########################################

    # Copying the images and the labels to the final folder
    pexels_sets = os.listdir(path_pexels_dataset)
    pexels_names = []
    for s in pexels_sets:
        path_pexels_annotations = os.path.join(path_pexels_dataset, s, "annotations", "obj_train_data")
        path_pexels_images = os.path.join(path_pexels_dataset, s, "images")
        pexels_labels = os.listdir(path_pexels_annotations)
        pexels_names.extend([os.path.splitext(l)[0] for l in pexels_labels])

        for l in pexels_labels:
            img_name = os.path.splitext(l)[0] + ".jpg"
            img_source = os.path.join(path_pexels_images, img_name)
            img_dest = os.path.join(path_processed_images, img_name)
            shutil.copy(img_source, img_dest)

            label_source = os.path.join(path_pexels_annotations, l)
            label_dest = os.path.join(path_processed_labels, l)
            shutil.copy(label_source, label_dest)

    # Separate the Helen dataset in training and validation
    train_pct = 0.7
    random.shuffle(pexels_names)
    train_size = int(train_pct*len(pexels_names))
    pexels_train_names = pd.DataFrame({0: pexels_names[:train_size]})
    pexels_val_names = pd.DataFrame({0: pexels_names[train_size:]})

    ###################################################################
    # PART 3: PROCESSING THE AFW IMAGES (Annotated Faces in the Wild) #
    ###################################################################

    # Make train/val splits
    afw_images = glob.glob(os.path.join(path_afw_dataset, "*.jpg"))
    afw_names = list(set([os.path.basename(f).split("_")[0] for f in afw_images]))  # removes duplicates
    random.shuffle(afw_names)
    train_size = int(train_pct * len(afw_names))
    afw_train_names = pd.DataFrame({0: afw_names[:train_size]})
    afw_val_names = pd.DataFrame({0: afw_names[train_size:]})

    # I will have to convert the 68 landmarks to the YOLO format as in the Helen dataset
    part_points_afw = {'jaw': [list(range(0, 17))],
                       'eye': [list(range(36, 42)), list(range(42, 48))],  # left and right
                       'nose': [list(range(27, 36)) + [21, 22]],
                       'mouth': [list(range(48, 68))],
                       'eyebrow': [list(range(17, 22)), list(range(22, 27))]}  # left and right
    for n in afw_names:
        grouped_images = glob.glob(os.path.join(path_afw_dataset, "{}*.jpg".format(n)))
        grouped_points = glob.glob(os.path.join(path_afw_dataset, "{}*.pts".format(n)))

        for i, img_name in enumerate(grouped_images):
            # For images with more than one face, the images are named like this:
            #   - 18489332_1.jpg
            #   - 18489332_2.jpg
            # We just want a single image (18489332.jpg)
            img_source = os.path.join(path_afw_dataset, img_name)
            img_dest = os.path.join(path_processed_images, "{}.jpg".format(n))
            shutil.copy(img_source, img_dest)

            with open(os.path.join(path_afw_dataset, grouped_points[i])) as f:
                lines = [l.rstrip() for l in f.readlines()][3:-1]  # keeping just the important lines

            # Processing the labels
            if i == 0:
                img = cv2.imread(os.path.join(img_source), cv2.IMREAD_COLOR)
                img_h, img_w = img.shape[:2]
                img, ratio = smart_resize(img)

                cv2.imshow("Image", img)
                cv2.waitKey(IMSHOW_WAIT_TIME)
                print("{}: {} landmarks".format(n, len(lines)))

            img_labels = pd.DataFrame(columns=['class', 'x', 'y', 'w', 'h'])
            for part_id, part_name in enumerate(use_parts):
                for idxs in part_points_afw[part_name]:
                    points = [lines[j] for j in idxs]
                    contour = []
                    for _, p in enumerate(points):
                        x, y = [float(c) for c in p.split(" ")]
                        contour.append([x, y])
                        img = cv2.circle(img, (int(x * ratio), int(y * ratio)), 3, (0, 255, 255), -1)

                    # Getting the bounding box in YOLO format
                    x, y, w, h = points_to_YOLO(img_labels, contour, part_id, img_h, img_w)

                    # Showing the box
                    img = cv2.rectangle(img,
                                        (int(x*ratio), int(y*ratio)),
                                        (int((x+w)*ratio), int((y+h)*ratio)), (0, 0, 255), 2)

            cv2.imshow("Image", img)
            cv2.waitKey(IMSHOW_WAIT_TIME)

            label_dest = os.path.join(path_processed_labels, "{}.txt".format(n))
            img_labels.round(6).to_csv(label_dest, header=False, index=False, sep=" ")

    #########################################
    # PART 4: PROCESSING THE MENPO2D IMAGES #
    #########################################

    # Menpo2D is already split into train/val
    menpo2D_split_data = {}
    for s in ['Train', 'Test']:
        split_dict = {'images': [],
                      'landmarks': []}
        for img_type in ['profile', 'semifrontal']:
            with open(os.path.join(path_menpo2D_dataset, s, "Menpo2D_{}_{}.txt".format(img_type, s.lower()))) as f:
                lines = [l.rstrip() for l in f.readlines()]
                split_menpo2D_images = [l.rstrip().split(" ")[0] for l in lines]
                split_menpo2D_landmarks = [l.rstrip().split(" ")[1:] for l in lines]
                split_dict['images'].extend(split_menpo2D_images)
                split_dict['landmarks'].extend(split_menpo2D_landmarks)

        menpo2D_split_data[s.lower()] = split_dict

    # Train and test names (for the YAML file)
    menpo2D_train_names = [os.path.splitext(os.path.basename(f))[0] for f in menpo2D_split_data['train']['images']]
    menpo2D_test_names = [os.path.splitext(os.path.basename(f))[0] for f in menpo2D_split_data['test']['images']]
    menpo2D_train_names = pd.DataFrame({0: menpo2D_train_names})
    menpo2D_test_names = pd.DataFrame({0: menpo2D_test_names})

    # I will have to convert the 68 landmarks to the YOLO format as in the Helen dataset
    part_points_menpo = {'semifrontal': part_points_afw,
                         'profile': {'jaw': [list(range(0, 12))],
                                     'eye': [list(range(22, 27))],
                                     'nose': [list(range(16, 22))],
                                     'mouth': [list(range(27, 39))],
                                     'eyebrow': [list(range(12, 16))]}}

    for split_name, split_data in menpo2D_split_data.items():
        split_images, split_landmarks = split_data['images'], split_data['landmarks']

        for i, img_path in enumerate(split_images):
            img_source = os.path.join(path_menpo2D_dataset, split_name.capitalize(), img_path)
            img_dest = os.path.join(path_processed_images, os.path.basename(img_path))
            shutil.copy(img_source, img_dest)

            # Detecting if the image is semifrontal (68 landmarks) or profile (39 landmarks)
            img_landmarks = [float(lmk) for lmk in split_landmarks[i]]
            x_points, y_points = img_landmarks[14::2], img_landmarks[15::2]
            img_type = 'profile' if len(x_points) == 39 else 'semifrontal'

            img = cv2.imread(os.path.join(img_source), cv2.IMREAD_COLOR)
            img_h, img_w = img.shape[:2]
            img, ratio = smart_resize(img)

            cv2.imshow("Image", img)
            cv2.waitKey(IMSHOW_WAIT_TIME)
            print("{}: {} landmarks".format(os.path.basename(img_path), len(x_points)))

            img_labels = pd.DataFrame(columns=['class', 'x', 'y', 'w', 'h'])
            for part_id, part_name in enumerate(use_parts):
                for idxs in part_points_menpo[img_type][part_name]:
                    x_points_part = [x_points[j] for j in idxs]
                    y_points_part = [y_points[j] for j in idxs]
                    contour = []

                    for x, y in zip(x_points_part, y_points_part):
                        contour.append([x, y])
                        img = cv2.circle(img, (int(x * ratio), int(y * ratio)), 3, (0, 255, 255), -1)

                    # Getting the bounding box in YOLO format
                    x, y, w, h = points_to_YOLO(img_labels, contour, part_id, img_h, img_w)

                    # Showing the box
                    img = cv2.rectangle(img,
                                        (int(x*ratio), int(y*ratio)),
                                        (int((x+w)*ratio), int((y+h)*ratio)), (0, 0, 255), 2)

            cv2.imshow("Image", img)
            cv2.waitKey(IMSHOW_WAIT_TIME)

            label_name = os.path.splitext(os.path.basename(img_path))[0]
            label_dest = os.path.join(path_processed_labels, "{}.txt".format(label_name))
            img_labels.round(6).to_csv(label_dest, header=False, index=False, sep=" ")

    ##################################
    # PART 5: CREATING THE YAML FILE #
    ##################################

    # Using the original Helen splits (test will be used for validation) and adding the Pexels and AFW splits
    skip_helen_ids = [os.path.splitext(s)[0] for s in skip_imgs]

    train_names = pd.read_csv(os.path.join(path_helen_dataset, 'trainnames.txt'), header=None)
    train_names = pd.concat([train_names,
                             pexels_train_names,
                             afw_train_names,
                             menpo2D_train_names], ignore_index=True)
    process_names(train_names, "train", path_processed_dataset, path_yolo_data, skip_helen_ids)

    test_names = pd.read_csv(os.path.join(path_helen_dataset, 'testnames.txt'), header=None)
    test_names = pd.concat([test_names,
                            pexels_val_names,
                            afw_val_names,
                            menpo2D_test_names], ignore_index=True)
    process_names(test_names, "val", path_processed_dataset, path_yolo_data, skip_helen_ids)

    # Creating the YAML file for training
    # Make sure that the class IDs are the same for all datasets! (i.e. 'eye' is class 0 in all datasets)
    with open(os.path.join(path_yolo_data, 'data.yaml'), 'w') as f:
        data = {'path': str(path_yolo_data),
                'train': os.path.join("images", "train", "images.txt"),
                'val': os.path.join("images", "val", "images.txt"),
                'test': '',
                'names': {i: p for i, p in enumerate(use_parts)}}
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    print("Done!")
