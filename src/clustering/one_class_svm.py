import numpy as np
import pandas as pd

from sklearn import svm
from sklearn.metrics import accuracy_score
from settings import NORMAL_FEATURE_FILE, ABNORMAL_FEATURE_FILE
from sklearn.model_selection import train_test_split

print("Import the whole features of normal images...")
x_train_total = pd.read_csv(NORMAL_FEATURE_FILE, header=None)


def detect_abnormal_from_one_class_svm(tmp):
    """
    Train one-class svm model with normal image sub-data and detect abnormal images from normal and abnormal images.
    :param tmp: number of sub-normal images to train.
    :return:
    """

    print("Import the features of normal and abnormal images...")
    x_train = x_train_total.sample(n=tmp)
    x_train.to_csv("cropped.csv", index=False, header=None)
    x_test = pd.read_csv(ABNORMAL_FEATURE_FILE, header=None)

    clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)

    print("Train model with selected normal images...")

    clf.fit(x_train.to_numpy())

    print("Predict images with model...")
    y_pred_train = clf.predict(x_train.to_numpy())
    y_pred_test = clf.predict(x_test.to_numpy())

    n_error_normal = y_pred_train[y_pred_train == -1].size
    n_error_abnormal = y_pred_test[y_pred_test == -1].size

    print("Abnormality of {} Normal Images".format(tmp), n_error_normal)
    print("Abnormality of 31 Abnormal Images", n_error_abnormal)

    return n_error_normal, n_error_abnormal


def estimate_inception_feature():

    x_normal = pd.read_csv("../../cropped.csv").to_numpy()
    x_abnormal = pd.read_csv(ABNORMAL_FEATURE_FILE).to_numpy()
    y_normal_label = np.ones(999)
    y_abnormal_label = np.zeros(30)

    x_data = np.concatenate((x_normal, x_abnormal), axis=0)
    y_data = np.concatenate((y_normal_label, y_abnormal_label), axis=0)

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=1234123)
    svm_model = svm.SVC(kernel='rbf')
    svm_model.fit(x_train, y_train)

    y_pred = svm_model.predict(x_abnormal)
    print(y_pred)
    accuracy = accuracy_score(y_abnormal_label, y_pred)

    print(accuracy)


if __name__ == '__main__':

    detect_abnormal_from_one_class_svm(tmp=15000)
