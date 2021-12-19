from sklearn import linear_model
from sklearn import ensemble
from sklearn import metrics


class Models:
    def __init__(self):
        self.classes = {"Linear regression": linear_model.LinearRegression,
                        "Gradient Boosting regression": ensemble.GradientBoostingRegressor,
                        "Logistic regression": linear_model.LogisticRegression,
                        "Gradient Boosting classifier": ensemble.GradientBoostingClassifier}
        self.metrics = {"accuracy": metrics.accuracy_score,
                        "mse": metrics.mean_squared_error,
                        "mae": metrics.mean_absolute_error,
                        "f1": metrics.f1_score,
                        "precision": metrics.precision_score,
                        "recall": metrics.recall_score}
