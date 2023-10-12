"""
This script puts the V2 and V3 folders of the FASSEG dataset into a single folder
I need to do this before uploading the images to CVAT.

Author: Ignacio Hernández Montilla, 2023
"""

import shutil
from pathlib import Path
import pandas as pd


if __name__ == "__main__":

    # I will annotate the V2 and V3 subsets of FASSEG

    path_fasseg = Path.home() / "Documents" / "Datasets" / "FASSEG"
    path_joined_images = path_fasseg / "joined-V2-V3" / "images"
    path_joined_images.mkdir(exist_ok=True, parents=True)

    split_info = pd.DataFrame(columns=['image', 'subset', 'is_train'])
    for v in [2, 3]:
        path_fasseg_v = path_fasseg / "V{}".format(v)
        images = list(path_fasseg_v.glob("*_RGB/*.bmp" if v == 2 else "*_RGB/**/*.bmp"))
        print("{}: {} images".format(path_fasseg_v, len(images)))
        for f in images:
            shutil.copy(f, path_joined_images / f.name)
            split_info.loc[len(split_info), :] = [f.name, v, int("Train_" in str(f))]

    split_info.to_csv(path_joined_images.parent / "split_info.csv", index=False)
