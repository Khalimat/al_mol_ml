from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import pandas as pd

from .utilities import calc_auc_ci


class Validation:
    def __init__(self, model, X, Y, name):
        self.model = model
        self.X = X
        self.Y = Y
        self.name = name
        self.results = None
        self.run()

    @staticmethod
    def f_one_mcc_score(model, X_test, Y_test):
        y_pred = model.predict(X_test)
        f_one = f1_score(Y_test, y_pred)
        tn, fp, fn, tp = confusion_matrix(Y_test, y_pred).ravel()
        return (
            f_one,
            (tp * tn - fp * fn)
            / ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5,
        )

    def validate(self, model, X_test, Y_test):
        test_predicted = model.predict_proba(X_test)
        auc, (lb_auc, ub_auc) = calc_auc_ci(Y_test, test_predicted[:, 1])
        f_one, mcc = self.f_one_mcc_score(model, X_test, Y_test)
        accuracy = accuracy_score(Y_test, self.model.predict(X_test))
        performance_stats = [lb_auc, auc, ub_auc, accuracy, f_one, mcc]
        return performance_stats

    def run(self):
        test_stats = self.validate(self.model, self.X, self.Y)
        results = pd.DataFrame(
            [test_stats],
            index=[self.name],
            columns=[
                "AUC lower estimate",
                "AUC",
                "AUC upper estimate",
                "accuracy",
                "F1",
                "MCC",
            ],
        )
        self.results = results
