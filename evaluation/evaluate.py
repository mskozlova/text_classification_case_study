import json
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


def generate_sla_for_tags(tags, file_name="sla.json"):
    sla_data = {"tag_sla": {t: (10, random.randint(20, 30)) for t in tags}}
    sla_data["manual_label_sla"] = (3, 10)
    sla_data["change_queues_penalty"] = (5, 10)
    
    with open(file_name, "w", encoding="utf8") as file:
        json.dump(
            sla_data,
            file,
            ensure_ascii=False,
            indent=4
        )


class WaitingTimeEvaluator:
    def __init__(self, file_name):
        with open(file_name, "r") as file:
            self.sla_data = json.load(file)
            
    
    def _estimate_waiting_time(self, true_label, pred_label, is_auto_reply):
        if is_auto_reply:
            assert pred_label != "UNK", "Can't auto reply with label UNK"
            return 0

        if true_label == pred_label:
            return random.randint(*self.sla_data["tag_sla"][true_label])
        
        if pred_label == "UNK":
            return random.randint(*self.sla_data["tag_sla"][true_label]) + \
                random.randint(*self.sla_data["manual_label_sla"])
        
        return random.randint(*self.sla_data["tag_sla"][true_label]) + \
            random.randint(*self.sla_data["manual_label_sla"]) + \
            random.randint(*self.sla_data["change_queues_penalty"])
    
    
    def __call__(self, true_labels, pred_labels, is_auto_reply=None, method="test"):
        assert len(true_labels) == len(pred_labels), \
            "true_labels and pred_labels have different lengths: {}, {}".format(len(true_labels), len(pred_labels))

        if is_auto_reply is None:
            is_auto_reply = [False] * len(true_labels)

        assert len(true_labels) == len(is_auto_reply), \
            "true_labels and is_automatic have different lengths: {}, {}".format(len(true_labels), len(is_auto_reply))
        
        data = pd.DataFrame({"true_label": true_labels, "pred_label": pred_labels, "is_auto_reply": is_auto_reply})
        data["method"] = method
        
        data["waiting_time"] = data.apply(
            lambda row: self._estimate_waiting_time(row["true_label"], row["pred_label"], row["is_auto_reply"]),
            axis=1
        )
        return data
