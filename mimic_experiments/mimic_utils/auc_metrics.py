# code by https://github.com/YerevaNN/mimic3-benchmarks/blob/220565b5ea3552ae487b41b6dd862f3a619f7619/mimic3models/metrics.py

import numpy as np
from sklearn import metrics


def print_metrics_multilabel(y_true, predictions, verbose=1):
    y_true = np.array(y_true)
    predictions = np.array(predictions)

    auc_scores = metrics.roc_auc_score(y_true, predictions, average=None)
    ave_auc_micro = metrics.roc_auc_score(y_true, predictions, average="micro")
    ave_auc_macro = metrics.roc_auc_score(y_true, predictions, average="macro")
    ave_auc_weighted = metrics.roc_auc_score(y_true, predictions, average="weighted")

    if verbose:
        print("ROC AUC scores for labels:", auc_scores)
        print("ave_auc_micro = {}".format(ave_auc_micro))
        print("ave_auc_macro = {}".format(ave_auc_macro))
        print("ave_auc_weighted = {}".format(ave_auc_weighted))

    return {
        "auc_scores": auc_scores,
        "ave_auc_micro": ave_auc_micro,
        "ave_auc_macro": ave_auc_macro,
        "ave_auc_weighted": ave_auc_weighted,
    }
