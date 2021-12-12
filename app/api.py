from flask import Flask, request, jsonify
from celery import Celery
from pymongo import MongoClient
import os

CELERY_BROKER = os.environ["CELERY_BROKER"]
CELERY_BACKEND = os.environ["CELERY_BACKEND"]
celery = Celery("tasks", broker=CELERY_BROKER, backend=CELERY_BACKEND)


app = Flask(__name__)

client = MongoClient("mongodb", 27017)
db = client["database"]
collection = db['model_collection']


@app.route('/')
def hello_world():
    return 'This is ML deploy app'


@app.route('/classes', methods=["GET"])
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
def create_model():
    """
    Function creates new model with specified name,class and hyperparameters from request
    :return: message that model was created
    """
    name = request.args.get("name")
    class_name = request.args.get("class")
    params = request.args.get("params")
    task = celery.send_task("create", args=[name, class_name, params])
    return task.id, 200


@app.route('/delete', methods=['GET'])
def delete():
    """
    Function deletes model with the given name
    :return: message that model has been deleted
    """
    name = request.args.get("name")
    task = celery.send_task("delete", args=[name])
    return task.id, 200


@app.route('/train', methods=['GET'])
def train():
    """
    Function gets name of the model and training data from request and trains the model
    :return: Message that training was successful
    """
    data = request.get_json()
    name = request.args.get("name")
    task = celery.send_task("train", args=[data, name])
    return task.id, 200


@app.route('/predict', methods=['GET'])
def predict():
    """
    Function gets data and model name from request,
    then loads model and uses it for predictions
    :return: predictions of the model in float type
    """
    data = request.get_json()
    name = request.args.get("name")
    task = celery.send_task("predict", args=[data, name])
    return task.id, 200


@app.route('/results', methods=['GET'])
def results():
    """
    Function returns results of a tak by given task_id
    :return: results of a task, or it's state
    """
    task_id = request.args.get("task_id")
    res = celery.AsyncResult(task_id)
    if res.state == "SUCCESS":
        return jsonify(res.get()), 200
    else:
        return str(res.state), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
