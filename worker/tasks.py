from celery import Celery
from models import Models
import pandas as pd
import os
import json
import mlflow
import logging

CELERY_BROKER = os.environ["CELERY_BROKER"]
CELERY_BACKEND = os.environ["CELERY_BACKEND"]
celery = Celery("tasks", broker=CELERY_BROKER, backend=CELERY_BACKEND, task_eager_propagates=True)

logger = logging.getLogger("worker")
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s:%(levelname)s:%(message)s")

file_handler = logging.FileHandler("/logs/worker.log")
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)


@celery.task(name="classes")
def classes():
    """
    Gets available model classes
    :return: dict with a list of model classes
    """

    models_dict = Models()
    logger.debug("Getting available classes")
    return {"classes": list(models_dict.classes.keys())}


@celery.task(name="models")
def models():
    """
    Returns models stored in database
    :return: dict with available models in database
    """
    logger.debug("Getting available models")
    client = mlflow.tracking.MlflowClient()
    models_dict = {"models": []}
    for rm in client.list_registered_models():
        models_dict['models'].append(dict(rm)['name'])
    return models_dict


@celery.task(name="delete")
def delete(name, version):
    """
    deletes model from a database
    :param name: name of the model to delete
    :param version:
    :return: dict with available models in database
    """
    logger.debug(f"Deleting model named{name}")
    client = mlflow.tracking.MlflowClient()
    try:
        if version:
            client.delete_model_version(name=name, version=version)
        else:
            client.delete_registered_model(name=name)
        logger.debug(f"Model {name} has been deleted")
        models_dict = {"models": []}
        for rm in client.list_registered_models():
            models_dict['models'].append(dict(rm)['name'])
        return models_dict
    except TypeError:
        logger.exception("Model with this name doesn't exist")


@celery.task(name="create")
def create(name, class_name, params):
    """
    creates new model and stores it in a database
    :param name: name of the model
    :param class_name: class of the model
    :param params: hyperparameters of the model
    :return: message that model has been created
    """
    logger.debug(f"Creating model with name:{name}, class: {class_name}, params: {params}")
    models_dict = Models()
    if params:
        params = json.loads(params)
        try:
            model = models_dict.classes[class_name](**params)
        except KeyError:
            logger.exception("Wrong class for model creation")
            raise KeyError("Wrong class for model creation")
    else:
        try:
            model = models_dict.classes[class_name]()
        except KeyError:
            logger.exception("Wrong class for model creation")
            raise KeyError("Wrong class for model creation")
    mlflow.start_run(run_name=name)
    mlflow.sklearn.log_model(model, "models/"+name, registered_model_name=name)
    mlflow.end_run()
    logger.debug(f"model named {name} has been created")
    return f"Model {name} has been created"


@celery.task(name="train")
def train(data, name, version):
    """
    Trains the model on the given data and saves model to database
    :param data: training data, should have target column
    :param name: name of the model to train
    :param version:
    :return: message that model has been train
    """
    data = pd.DataFrame.from_dict(data)
    logger.debug(f"Training model with name:{name}, data: {data.dtypes}")
    mlflow.start_run()
    model = mlflow.sklearn.load_model(
        model_uri=f"models:/{name}/{version}"
    )
    try:
        x = data.drop(columns="target")
        y = data["target"]
        model.fit(x, y)
        mlflow.sklearn.log_model(model, "model", registered_model_name=name)
        mlflow.end_run()
        return "Training successful", 200
    except AttributeError:
        mlflow.end_run()
        logger.exception("Model with the given name doesn't exist")
        raise AttributeError("Model with the given name doesn't exist")
    except KeyError:
        mlflow.end_run()
        logger.exception("No 'target' column in data")
        raise KeyError("No 'target' column in data")


@celery.task(name="test")
def test(data, name, version, metric):
    """
    Function that predicts target values for the data with a given model
    :param data: data for prediction
    :param name: name of the model
    :param version:
    :return: list of predictions
    """
    data = pd.DataFrame.from_dict(data)
    models_dict = Models()
    model = mlflow.sklearn.load_model(
        model_uri=f"models:/{name}/{version}"
    )
    logger.debug(f"Testing with parameters: Name:{name}, version: {version}, metric: {metric}, data {data.dtypes}")
    try:
        x = data.drop(columns="target")
        y = data["target"]
        predictions = model.predict(x)
        score = models_dict.metrics[metric](y, predictions)
        mlflow.start_run()
        mlflow.log_metric(metric, score)
        mlflow.sklearn.log_model(model, "models/" + name, registered_model_name=name)
        mlflow.end_run()
        return {"Test "+metric: score}
    except AttributeError:
        mlflow.end_run()
        logger.exception("Model with the given name doesn't exist")


@celery.task(name="predict")
def predict(data, name, version):
    """
    Function that predicts target values for the data with a given model
    :param data: data for prediction
    :param name: name of the model
    :param version:
    :return: list of predictions
    """
    data = pd.DataFrame.from_dict(data)
    model = mlflow.sklearn.load_model(
        model_uri=f"models:/{name}/{version}"
    )
    logger.debug(f"Predicting with parameters: Name:{name}, version: {version, }data {data.dtypes}")
    try:
        predictions = model.predict(data)
        return {"predictions": predictions.tolist()}
    except AttributeError:
        logger.exception("Model with the given name doesn't exist")
