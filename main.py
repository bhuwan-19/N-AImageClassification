import numpy as np

from src.clustering.one_class_svm import detect_abnormal_from_one_class_svm
from matplotlib import pyplot as plt
from settings import START_TMP, END_TMP, TMP_INTERVAL


if __name__ == '__main__':

    normal_errs = []
    abnormal_errs = []

    for i in range(START_TMP, END_TMP, TMP_INTERVAL):

        normal_err, abnormal_err = detect_abnormal_from_one_class_svm(tmp=i)
        normal_errs.append(normal_err)
        abnormal_errs.append(abnormal_err)

    x = np.arange(START_TMP, END_TMP, TMP_INTERVAL)
    normal_errs = np.array(normal_errs)
    abnormal_errs = np.array(abnormal_errs)
    plt.subplot(121), plt.plot(x, normal_errs), plt.title("Number of Abnormality of Normal Images")
    plt.subplot(122), plt.plot(x, abnormal_errs), plt.title("Number of Abnormality of Abnormal Images")
    plt.show()
