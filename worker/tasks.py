from bson import Binary
from bson.objectid import ObjectId
from celery import Celery
from pymongo import MongoClient
from models import Models
import pandas as pd
import os
import json
import pickle

CELERY_BROKER = os.environ["CELERY_BROKER"]
CELERY_BACKEND = os.environ["CELERY_BACKEND"]
celery = Celery("tasks", broker=CELERY_BROKER, backend=CELERY_BACKEND)

client = MongoClient("mongodb", 27017)
db = client["database"]
collection = db['model_collection']


@celery.task(name="classes")
def classes():
    """
    Gets available model classes
    :return: dict with a list of model classes
    """
    models_dict = Models()
    return {"classes": list(models_dict.classes.keys())}


@celery.task(name="models")
def models():
    """
    Returns models stored in database
    :return: dict with available models in database
    """
    return {model["name"]: {"class": model["class"], "Trained": model["trained"]} for model in collection.find({})}


@celery.task(name="delete")
def delete(name):
    """
    deletes model from a database
    :param name: name of the model to delete
    :return: dict with available models in database
    """
    query = collection.find_one({"name": name})
    collection.delete_one({'_id': ObjectId(query["_id"])})
    return {model["name"]: {"class": model["class"], "Trained": model["trained"]} for model in collection.find({})}


@celery.task(name="create")
def create(name, class_name, params):
    """
    creates new model and stores it in a database
    :param name: name of the model
    :param class_name: class of the model
    :param params: hyperparameters of the model
    :return: message that model has been created
    """
    models_dict = Models()
    if params:
        params = json.loads(params)
        try:
            model = models_dict.classes[class_name](**params)
        except KeyError:
            raise KeyError("Wrong class")
    else:
        try:
            model = models_dict.classes[class_name]()
        except KeyError:
            raise KeyError("Wrong class")
    model = pickle.dumps(model)
    collection.insert_one({"name": name, "class": class_name, "trained": False, "model": Binary(model)})
    return f"Model {name} has been created"


@celery.task(name="train")
def train(data, name):
    """
    Trains the model on the given data and saves model to database
    :param data: training data, should have target column
    :param name: name of the model to train
    :return: message that model has been train
    """
    data = pd.DataFrame.from_dict(data)
    model = collection.find_one({"name": name})
    model = pickle.loads(model["model"])
    try:
        x = data.drop(columns="target")
        y = data["target"]
        model.fit(x, y)
        model = pickle.dumps(model)
        collection.update_one({"name": name}, {"$set": {"trained": True,
                                                        "model": Binary(model)}})
        return "Training successful", 200
    except AttributeError:
        raise AttributeError("Model with the given name doesn't exist")
    except KeyError:
        raise KeyError("No 'target' column in data")


@celery.task(name="predict")
def predict(data, name):
    """
    Function that predicts target values for the data with a given model
    :param data: data for prediction
    :param name: name of the model
    :return: list of predictions
    """
    data = pd.DataFrame.from_dict(data)
    model = collection.find_one({"name": name})["model"]
    model = pickle.loads(model)
    try:
        predictions = model.predict(data)
        return {"predictions": predictions.tolist()}
    except AttributeError:
        raise AttributeError("Model with the given name doesn't exist")
