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
        """
        :return: list of available models in models directory
        """
        available_models = [i for i in os.listdir(PATH) if i.endswith('.pickle')]
        return available_models

    def get(self, name):
        """
        Loads the model with the given name from pickle
        :param name: model name
        :return: model
        """
        if f'{name}.pickle' in self.get_models_list():
            with open(file=f'{PATH}/{name}.pickle', mode='rb') as f:
                return pickle.load(f)

    def create(self, class_name, name, params):
        """
        Creates and saves new model
        :param class_name:class of new model
        :param name:name of new model
        :param params:hyperparameters of new model
        :return: name of created model
        """
        if not name:
            name = self.counter
            self.counter += 1
        if params:
            try:
                with open(file=f'{PATH}/{self.counter}.pickle', mode='wb') as f:
                    pickle.dump(self.classes[class_name](**params), f)
            except KeyError:
                raise KeyError("Wrong class")
        else:
            try:
                with open(file=f'{PATH}/{self.counter}.pickle', mode='wb') as f:
                    pickle.dump(self.classes[class_name](), f)
            except KeyError:
                raise KeyError("Wrong class")
        return name

    def train(self, name, data):
        """
        Trains the model on the data and then saves model in the models directory
        :param name: name of the model to train
        :param data: data for the model to train on, requires to have column "target"
        :return:
        """

        model = self.get(name)
        try:
            x = data.drop(columns="target")
            y = data["target"]
            model.fit(x, y)
            with open(file=f'{PATH}/{name}.pickle', mode='wb') as f:
                pickle.dump(model, f)
        except AttributeError:
            raise AttributeError("Model with the given name doesn't exist")
        except KeyError:
            raise KeyError("No 'target' column in data")

    def delete_model(self, name):
        """
        Deletes model with the given name from models directory
        :param name: name of model to delete
        :return:
        """
        if f'{name}.pickle' in self.get_models_list():
            os.remove(f'{PATH}/{name}.pickle')
