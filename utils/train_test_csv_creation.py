import os
import pandas as pd
import numpy as np

from settings import NORMAL_FEATURE_FILE, ABNORMAL_FEATURE_FILE, FEATURE_DIR


def create_train_test_csv():

    normal_features = pd.read_csv(NORMAL_FEATURE_FILE, header=None)
    abnormal_features = pd.read_csv(ABNORMAL_FEATURE_FILE, header=None)

    print("loaded normal and abnormal features...")

    # features = [normal_features, abnormal_features]
    # total_features = pd.concat(features).reset_index(drop=True)

    msk = np.random.rand(len(normal_features)) < 0.99
    train_data = normal_features[msk]

    test_normal_features = normal_features[~msk]
    print(len(test_normal_features))
    test_features = [test_normal_features, abnormal_features]
    test_data = pd.concat(test_features).reset_index(drop=True)
    # train_label = create_label(data_frame=train_data, base_val=len(normal_features))
    test_label = create_label(data_frame=test_data, base_val=len(test_normal_features))
    print("train data length:", len(train_data))
    print("test data length:", len(test_data))

    train_data_file = os.path.join(FEATURE_DIR, 'train_feature.csv')
    # train_label_file = os.path.join(FEATURE_DIR, 'train_label.csv')
    test_data_file = os.path.join(FEATURE_DIR, 'test_feature.csv')
    test_label_file = os.path.join(FEATURE_DIR, 'test_label.csv')

    train_data.to_csv(train_data_file, index=False)
    test_data.to_csv(test_data_file, index=False)
    # train_label.to_csv(train_label_file, index=False)
    test_label.to_csv(test_label_file, index=False)
    print("save train and test data set")


def create_label(data_frame, base_val):

    label = []
    for idx in data_frame.index:

        if idx >= base_val:
            label.append(1)
        else:
            label.append(0)

    label_df = pd.DataFrame(label)

    return label_df


if __name__ == '__main__':

    create_train_test_csv()
