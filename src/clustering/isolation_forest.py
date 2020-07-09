import numpy as np
import matplotlib.pyplot as plt
import h2o
# import glob
import os

from h2o.estimators import H2OIsolationForestEstimator
from sklearn.metrics import roc_curve, precision_recall_curve, auc, confusion_matrix, f1_score
from settings import TRAIN_FILE, TEST_FILE, TEST_LABEL_FILE, ISO_FOREST_MODEL, RESULT_TXT, ISO_FOREST_N_TREE_L, \
    ISO_FOREST_N_TREE_STEP, ISO_FOREST_N_TREE_U
from utils.folder_file_manager import save_file


def train_iso_forest_model(data_frame, n_tree_number):

    seed = 12345
    n_trees = n_tree_number
    iso_forest = H2OIsolationForestEstimator(ntrees=n_trees, seed=seed)
    iso_forest.train(training_frame=data_frame)
    iso_path = h2o.save_model(model=iso_forest, path=ISO_FOREST_MODEL)
    print("save isolation forest model")

    return iso_path


def predict_abnormal(model_path, data_frame, test_label):

    # model_path = glob.glob(os.path.join(ISO_FOREST_MODEL, "*"))
    model = h2o.load_model(model_path)
    prediction = model.predict(data_frame)

    quartile = 0.95
    quantile_frame = prediction.quantile([quartile])

    threshold = quantile_frame[0, "predictQuantiles"]
    prediction["predicted_class"] = prediction["predict"] > threshold
    prediction["class"] = test_label

    return prediction


def get_auc(labels, scores):

    fpr, tpr, thresholds = roc_curve(labels, scores)
    auc_score = auc(fpr, tpr)

    return fpr, tpr, auc_score


def get_aucpr(labels, scores):

    precision, recall, th = precision_recall_curve(labels, scores)
    aucpr_score = np.trapz(recall, precision)

    return precision, recall, aucpr_score


def plot_metric(ax, x, y, x_label, y_label, plot_label, style="-"):

    ax.plot(x, y, style, label=plot_label)
    ax.legend()

    ax.set_ylabel(x_label)
    ax.set_xlabel(y_label)


def prediction_summary(labels, predicted_score, predicted_class, info, ntrees, plot_baseline=True, axes=None):

    p_txt = os.path.join(RESULT_TXT, 'n_tree_{}_precision.txt'.format(ntrees))
    r_txt = os.path.join(RESULT_TXT, 'n_tree_{}_recall.txt'.format(ntrees))
    aucpr_txt = os.path.join(RESULT_TXT, 'n_tree_{}_aucpr.txt'.format(ntrees))
    f1_txt = os.path.join(RESULT_TXT, 'n_tree_{}_f1.txt'.format(ntrees))
    conf_txt = os.path.join(RESULT_TXT, 'n_tree_{}_confusion.txt'.format(ntrees))

    if axes is None:
        axes = [plt.subplot(1, 2, 1), plt.subplot(1, 2, 2)]

    fpr, tpr, auc_score = get_auc(labels, predicted_score)
    plot_metric(axes[0], fpr, tpr, "False positive rate",
                "True positive rate", "{} AUC = {:.4f}".format(info, auc_score))
    if plot_baseline:
        plot_metric(axes[0], [0, 1], [0, 1], "False positive rate",
                    "True positive rate", "baseline AUC = 0.5", "r--")

    precision, recall, aucpr_score = get_aucpr(labels, predicted_score)
    f1 = f1_score(labels, predicted_class)
    conf_mat = confusion_matrix(labels, predicted_class)
    save_file(content=precision, filename=p_txt, method='w')
    save_file(content=recall, filename=r_txt, method='w')
    save_file(content=aucpr_score, filename=aucpr_txt, method='w')
    save_file(content=f1, filename=f1_txt, method='w')
    save_file(content=conf_mat, filename=conf_txt, method='w')

    plot_metric(axes[1], recall, precision, "Recall",
                "Precision", "{} AUCPR = {:.4f}".format(info, aucpr_score))
    if plot_baseline:
        thr = sum(labels) / len(labels)
        plot_metric(axes[1], [0, 1], [thr, thr], "Recall",
                    "Precision", "baseline AUCPR = {:.4f}".format(thr), "r--")

    plt.show()

    return axes


def figure(n_tree_n):

    fig_size = 4.5
    f = plt.figure()
    f.suptitle("ntrees={}".format(n_tree_n), fontsize=16)
    f.set_figheight(fig_size)
    f.set_figwidth(fig_size * 2)


if __name__ == '__main__':

    h2o.init()

    train_df = h2o.import_file(TRAIN_FILE)
    test_df = h2o.import_file(TEST_FILE)
    test_label_df = h2o.import_file(TEST_LABEL_FILE)

    for n_tree in range(ISO_FOREST_N_TREE_L, ISO_FOREST_N_TREE_U, ISO_FOREST_N_TREE_STEP):

        path = train_iso_forest_model(data_frame=train_df, n_tree_number=n_tree)

        predictions = predict_abnormal(model_path=path, data_frame=test_df, test_label=test_label_df)

        h2o_predictions = predictions.as_data_frame()

        figure(n_tree_n=n_tree)
        axes = prediction_summary(
            labels=h2o_predictions["class"], predicted_score=h2o_predictions["predict"],
            predicted_class=h2o_predictions["predicted_class"], info="h2o", ntrees=n_tree)
