from sklearn import linear_model
from sklearn import ensemble


class Models:
    def __init__(self):
        self.classes = {"Linear regression": linear_model.LinearRegression,
                        "Gradient Boosting regression": ensemble.GradientBoostingRegressor,
                        "Logistic regression": linear_model.LogisticRegression,
                        "Gradient Boosting classifier": ensemble.GradientBoostingClassifier}
