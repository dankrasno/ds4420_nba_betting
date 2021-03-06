from typing import Any, List

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_conf_mat(y_true: Any, y_pred: Any, labels: List[str] = ["L", "W"]) -> None:
    # Credit to https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea
    cf_matrix = confusion_matrix(y_true, y_pred)
    cf_matrix_df = pd.DataFrame(cf_matrix, columns=labels, index=labels)
    group_names = ["True Neg", "False Pos", "False Neg", "True Pos"]
    group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
    group_percentages = [
        "{0:.2%}".format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)
    ]
    annotations: np.ndarray[Any, np.dtype[np.str_]] = np.asarray(
        [
            f"{name}\n{count}\n{pct}"
            for name, count, pct in zip(group_names, group_counts, group_percentages)
        ]
    ).reshape(2, 2)
    sns.heatmap(
        cf_matrix_df,
        annot=annotations,
        fmt="",
        cmap="Blues",
    )
