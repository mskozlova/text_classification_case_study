import numpy as np
import pandas as pd
import random

from sklearn.metrics import precision_recall_curve


def get_class_threshold(y_true, y_pred_prob, target_class_precision=0.95):
    if len(y_true) == 0:
        return 1.1
    
    if sum(y_true) == 0:
        return 1.1
    
    precision, _, thresholds = precision_recall_curve(y_true, y_pred_prob)
    idxs = np.where(precision > target_class_precision)[0]
    
    if len(idxs) == 0 or idxs[0] >= len(thresholds):
        return 1.1
    
    return thresholds[idxs[0]]


def get_multiclass_tresholds(y_true, y_pred_probs, labels, target_class_precision=0.95):
    class_thresholds = dict()
    labels_predicted = np.argmax(np.asarray(y_pred_probs), axis=1)
    max_probs = np.max(np.asarray(y_pred_probs), axis=1)
    
    for i, label in enumerate(labels):
        label_predicted_idxs = np.where(labels_predicted == i)[0]
        y_pred_probs_label = np.take(max_probs, label_predicted_idxs)
        y_true_label = np.take(y_true, label_predicted_idxs) == label
        y_true_label = y_true_label.astype(int)
        
        class_thresholds[label] = get_class_threshold(
            y_true_label,
            y_pred_probs_label,
            target_class_precision=target_class_precision
        )
    
    return class_thresholds


def predict_multiclass_by_thresholds(y_pred_probs, labels, thresholds, unknown_label="UNK"):
    winning_labels = [labels[i] for i in np.argmax(y_pred_probs, axis=1)]
    winning_probs = np.max(y_pred_probs, axis=1)
    
    result = []
    for label, prob in zip(winning_labels, winning_probs):
        if prob >= thresholds[label]:
            result.append(label)
        else:
            result.append(unknown_label)
    
    return result


def predict_labels(test_true_labels, test_pred_probs, eval_pred_probs, class_names, target_class_precision=0.95):
    thresholds = get_multiclass_tresholds(
        test_true_labels,
        test_pred_probs,
        class_names,
        target_class_precision
    )
    return predict_multiclass_by_thresholds(
        eval_pred_probs,
        class_names,
        thresholds
    )
