import os

from utils.folder_file_manager import make_directory_if_not_exists

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = make_directory_if_not_exists(os.path.join(CUR_DIR, 'data', 'model'))
ONE_CLASS_MODEL = os.path.join(MODEL_DIR, 'one_class_svm.sav')
ISO_FOREST_MODEL = os.path.join(MODEL_DIR, 'iso_forest')

FEATURE_DIR = make_directory_if_not_exists(os.path.join(CUR_DIR, 'data', 'features'))
NORMAL_FEATURE_FILE = os.path.join(FEATURE_DIR, 'train_normal_feature.csv')
ABNORMAL_FEATURE_FILE = os.path.join(FEATURE_DIR, 'train_abnormal_feature.csv')
TRAIN_FILE = os.path.join(FEATURE_DIR, 'train_feature.csv')
TRAIN_LABEL_FILE = os.path.join(FEATURE_DIR, 'train_label.csv')
TEST_FILE = os.path.join(FEATURE_DIR, 'test_feature.csv')
TEST_LABEL_FILE = os.path.join(FEATURE_DIR, 'test_label.csv')

IMAGE_DIR = make_directory_if_not_exists(os.path.join(CUR_DIR, '../data', 'image'))
NORMAL_IMAGE_DIR = os.path.join(IMAGE_DIR, 'normal')
ABNORMAL_IMAGE_DIR = os.path.join(IMAGE_DIR, 'anormal')
RESULT_TXT = os.path.join(CUR_DIR, 'report')

INCEPTION_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
ALLOWED_EXTS = [".png", ".jpg"]

ONE_CLASS_START_TMP = 500
ONE_CLASS_END_TMP = 8000
ONE_CLASS_TMP_INTERVAL = 500

ISO_FOREST_N_TREE_L = 100
ISO_FOREST_N_TREE_U = 1000
ISO_FOREST_N_TREE_STEP = 100
