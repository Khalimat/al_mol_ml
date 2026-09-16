import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from .delong import calc_auc_ci
from .utilities import perf_columns


class Validation:
    """Computes AUC (with confidence interval), accuracy, F1, and MCC for
    `model` on a single (X, Y) set, as a one-row DataFrame indexed by `name`.
    """

    def __init__(self, model, X, Y, name):
        self.model = model
        self.X = X
        self.Y = Y
        self.name = name
        self.results = pd.DataFrame(
            [self._compute_metrics()], index=[name], columns=perf_columns
        )

    def _compute_metrics(self):
        y_proba = self.model.predict_proba(self.X)[:, 1]
        y_pred = self.model.predict(self.X)

        auc, (lb_auc, ub_auc) = calc_auc_ci(self.Y, y_proba)
        f1 = f1_score(self.Y, y_pred, zero_division=0)
        mcc = self._matthews_corrcoef(self.Y, y_pred)
        accuracy = accuracy_score(self.Y, y_pred)
        return [lb_auc, auc, ub_auc, accuracy, f1, mcc]

    @staticmethod
    def _matthews_corrcoef(y_true, y_pred):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        return (tp * tn - fp * fn) / ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
