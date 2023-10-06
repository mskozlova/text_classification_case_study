import numpy as np
from sklearn.metrics import confusion_matrix

import plotly.figure_factory as ff
import plotly.graph_objects as go


def draw_confusion_matrix(y_true, y_pred, classes_names, label, **kwargs):
    values = confusion_matrix(y_true, y_pred, **kwargs)

    x_labels, y_labels = list(classes_names), list(classes_names)
    values_text = [["{:.1f}%".format(val * 100) for val in row] for row in values]

    fig = ff.create_annotated_heatmap(
        values,
        x=x_labels, y=y_labels,
        annotation_text=values_text,
        colorscale="temps"
    )

    fig.update_layout(title_text="Confusion matrix, {}".format(label))
    fig.add_annotation({
        "font": dict(color="black", size=14),
        "x": 0.5, "y": -0.15,
        "showarrow": False,
        "text": "Predicted value",
        "xref": "paper", "yref": "paper"
    })
    fig.add_annotation({
        "font": dict(color="black", size=14),
        "x": -0.35, "y": 0.5,
        "showarrow": False,
        "text": "Real value",
        "textangle": -90,
        "xref": "paper", "yref": "paper"
    })


    fig.update_layout(margin={"t": 50, "l": 200})

    fig["data"][0]["showscale"] = True
    return fig


def show_lr_feature_importance(lr, class_idx, count_vectorizer):
    labels = list(map(lambda x: x[0], sorted(count_vectorizer.vocabulary_.items(), key=lambda x: x[1])))
    fi = lr.coef_[class_idx]

    features = sorted(zip(labels, fi), key=lambda x: x[1])[::-1]

    fig = go.Figure([go.Bar(
        x=list(map(lambda x: x[0], features)),
        y=list(map(lambda x: x[1], features))
    )])
    fig.update_layout(
        title="Baseline feature importances for label {}".format(lr.classes_[class_idx]),
        width=700,
        height=400
    )
    return fig


def show_predicted_labels_distribution(pred_labels, method):
    labels, counts = np.unique(pred_labels, return_counts=True)
    fig = go.Figure([go.Bar(x=labels, y=counts)])
    fig.update_layout(title="Predictions per tag {}".format(method), width=700, height=400)
    return fig
