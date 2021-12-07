from sklearn import linear_model
from sklearn import ensemble
import pickle
import os

PATH = "models"


class Models:
    def __init__(self):
        self.classes = {"Linear regression": linear_model.LinearRegression,
                        "Gradient Boosting regression": ensemble.GradientBoostingRegressor,
                        "Logistic regression": linear_model.LogisticRegression,
                        "Gradient Boosting classifier": ensemble.GradientBoostingClassifier}
        self.counter = 0

    @staticmethod
    def get_models_list():
        available_models = [i for i in os.listdir(PATH) if i.endswith('.pickle')]
        return available_models

    def get(self, name):
        if f'{name}.pickle' in self.get_models_list():
            with open(file=f'{PATH}/{name}.pickle', mode='rb') as f:
                return pickle.load(f)

    def create(self, class_name, name, params):
        if name:
            with open(file=f'{PATH}/{name}.pickle', mode='wb') as f:
                pickle.dump(self.classes[class_name](**params), f)
            return name
        else:
            with open(file=f'{PATH}/{self.counter}.pickle', mode='wb') as f:
                pickle.dump(self.classes[class_name](**params), f)
            self.counter += 1
            return self.counter - 1

    def train(self, name, data):
        model = self.get(name)
        x = data.drop(columns="target")
        y = data["target"]
        model.fit(x, y)
        with open(file=f'{PATH}/{name}.pickle', mode='wb') as f:
            return pickle.dump(model, f)

    def delete_model(self, name):
        if f'{name}.pickle' in self.get_models_list():
            os.remove(f'{PATH}/{name}.pickle')

