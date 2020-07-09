import glob
import os
import cv2

from settings import IMAGE_DIR
from utils.cv_utils import load_image


def collect_preprocessed_image():
    """
    Preprocess images and save them.
    :return:
    """

    image_paths = glob.glob(os.path.join(IMAGE_DIR, "*.png"))

    for path in image_paths:

        image = load_image(path=path, is_equ=True, is_norm=True, is_rotate=True)

        if image is not None:

            cv2.imwrite(path, image)

    return


if __name__ == '__main__':

    collect_preprocessed_image()
