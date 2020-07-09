import sys
import os
import glob
import csv

from src.feature_detection.image_feature import ImageFeature
from settings import NORMAL_IMAGE_DIR, ABNORMAL_IMAGE_DIR, FEATURE_DIR, NORMAL_FEATURE_FILE, ABNORMAL_FEATURE_FILE

inu = ImageFeature()


def collect_features(parsing_dir, save_file):
    """
    Collect the features of all the image in specific directory and save as a csv file.
    :param parsing_dir: directory which contains the images to extract feature.
    :param save_file: file path to save the features as a csv format
    :return:
    """

    sys.stdout.write(' >>> collect train data(features) from the raw images \n')
    if not os.path.isdir(parsing_dir):
        sys.stderr.write(" not exist folder for raw image data\n")
        sys.exit(1)

    """
    # --- check the raw images ----------------------------------------------------------
    raw_dir = IMAGE_DIR
    sub_dirs = []
    for child in os.listdir(raw_dir):
        child_path = os.path.join(raw_dir, child)
        if os.path.isdir(child_path):
            sub_dirs.append(child)
    sub_dirs.sort()
    labels = sub_dirs
    # 
    tails = []
    for i in range(len(sub_dirs)):
        line = np.zeros((len(sub_dirs)), dtype=np.uint8)
        line[i] = 1
        tails.append(line.tolist())
        
    
    tails = [[1., 0.],
             [0., 1.]]
    """

    # --- scanning the raw image dir ----------------------------------------------------
    image_paths = glob.glob(os.path.join(parsing_dir, "*.png"))
    features = []
    count = 0

    for path in image_paths:

        try:
            # Extract the feature vector per each image
            feature = inu.get_feature_from_image(path)
            sys.stdout.write("\r" + path)
            sys.stdout.flush()
        except Exception as e:
            print(e)
            continue
        line = feature.tolist()
        features.append(line)
        count += 1

        # if count > 10:  # for only testing
        #     break

    sys.stdout.write("\n counts #: {}\n".format(count))

    # --- write the train_data.csv file on the same location --------------------------------------
    save_dir = FEATURE_DIR
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    feature_data_path = save_file
    if sys.version_info[0] == 2:  # py 2x
        with open(feature_data_path, 'wb') as fp:  # for python 2x
            wr = csv.writer(fp, delimiter=',')
            wr.writerows(features)

    elif sys.version_info[0] == 3:  # py 3x
        with open(feature_data_path, 'w', newline='') as fp:  # for python 3x
            wr = csv.writer(fp, delimiter=',')
            wr.writerows(features)

    # write the train_label.txt on the same location
    # feature_label_path = os.path.join(save_dir, "train_label.txt")
    # with open(feature_label_path, 'w') as fp:
    #     for label in labels:
    #         fp.write(label + "\n") 1

    sys.stdout.write("create the train_data.csv successfully!\n")

    return save_dir


if __name__ == '__main__':

    collect_features(parsing_dir=NORMAL_IMAGE_DIR, save_file=NORMAL_FEATURE_FILE)
