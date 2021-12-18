from flask import Flask, request, jsonify
from prometheus_flask_exporter import PrometheusMetrics
from celery import Celery
from pymongo import MongoClient
import os
import logging

logging.basicConfig(filename="/logs/app.log",
                    format="%(asctime)s:%(levelname)s:%(message)s")

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.propagate = False

formatter = logging.Formatter("%(asctime)s:%(levelname)s:%(message)s")

file_handler = logging.FileHandler("/logs/app.log")
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)


CELERY_BROKER = os.environ["CELERY_BROKER"]
CELERY_BACKEND = os.environ["CELERY_BACKEND"]
celery = Celery("tasks", broker=CELERY_BROKER, backend=CELERY_BACKEND, task_eager_propagates=True)

logger.info("Starting APi")
app = Flask(__name__)
metrics = PrometheusMetrics(app)
metrics.info('app_info', 'Application info')

client = MongoClient("mongodb", 27017)
db = client["database"]
collection = db['model_collection']


@app.route('/')
def hello_world():
    return 'This is ML deploy app'


@app.route('/classes', methods=["GET"])
@metrics.histogram("hist_classes", "Classes requests by status code",
                   labels={'status': lambda r: r.status_code})
def get_classes():
    """
    :return: json with the list of model classes supported in this API
    """
    task = celery.send_task("classes")
    res = celery.AsyncResult(task.id)
    while True:
        if res.state == "SUCCESS":
            break
    return jsonify(res.get()), 200


@app.route('/models', methods=["GET"])
@metrics.histogram("hist_models", "Models requests by status code",
                   labels={'status': lambda r: r.status_code})
def get_models():
    """
    We use function get_models_list() of Models class instance to get the list of available models
    Names of the models shown with removed extension(.pickle)
    :return: json with the list of available models
    """
    task = celery.send_task("models")
    res = celery.AsyncResult(task.id)
    while True:
        if res.state == "SUCCESS":
            break
    return jsonify(res.get()), 200


@app.route('/create_model', methods=["GET"])
@metrics.counter("Count_model_creation", "Create requests by class",
                 labels={'class_name': lambda: request.args.get('class', 'none')})
def create_model():
    """
    Function creates new model with specified name,class and hyperparameters from request
    :return: message that model was created
    """
    name = request.args.get("name")
    class_name = request.args.get("class")
    params = request.args.get("params")
    logger.debug(f'Creating model with following parameters: Name:{name}, Class: {class_name}, params:{params}')
    task = celery.send_task("create", args=[name, class_name, params])
    return task.id, 200


@app.route('/delete', methods=['GET'])
def delete():
    """
    Function deletes model with the given name
    :return: message that model has been deleted
    """
    name = request.args.get("name")
    logger.debug(f'Deleting model with following parameters: Name:{name}')
    task = celery.send_task("delete", args=[name])
    return task.id, 200


@app.route('/train', methods=['GET'])
@metrics.counter("Count_training", "Training requests by model",
                 labels={'model': lambda: request.args.get('name', 'none')})
def train():
    """
    Function gets name of the model and training data from request and trains the model
    :return: Message that training was successful
    """
    data = request.get_json()
    name = request.args.get("name")
    logger.debug(f'Training model with following parameters: Name:{name}')
    task = celery.send_task("train", args=[data, name])
    return task.id, 200


@app.route('/predict', methods=['GET'])
@metrics.counter("Count_prediction", "Prediction requests by model",
                 labels={'model': lambda: request.args.get('name', 'none')})
def predict():
    """
    Function gets data and model name from request,
    then loads model and uses it for predictions
    :return: predictions of the model in float type
    """
    data = request.get_json()
    name = request.args.get("name")
    logger.debug(f'Predicting with following parameters: Name:{name}')
    task = celery.send_task("predict", args=[data, name])
    return task.id, 200


@app.route('/results', methods=['GET'])
@metrics.counter("Count_results", "Count results requests by status code",
                 labels={'status': lambda r: r.status_code})
def results():
    """
    Function returns results of a tak by given task_id
    :return: results of a task, or it's state
    """
    task_id = request.args.get("task_id")
    logger.debug(f'Getting results with following parameters: Task_id:{task_id}')
    res = celery.AsyncResult(task_id)
    if res.state == "SUCCESS":
        return jsonify(res.get()), 200
    elif res.state == "FAILURE":
        return str(res.state), 500
    else:
        return str(res.state), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
