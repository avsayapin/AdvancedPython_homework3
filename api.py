from flask import Flask, request, jsonify
from models import Models
import pandas as pd
import json

models = Models()

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'This is ML deploy app'


@app.route('/classes', methods=["GET"])
def get_classes():
    """
    :return: json with the list of model classes supported in this API
    """
    return jsonify(classes=list(models.classes.keys())), 200


@app.route('/models', methods=["GET"])
def get_models():
    """
    We use function get_models_list() of Models class instance to get the list of available models
    Names of the models shown with removed extension(.pickle)
    :return: json with the list of available models
    """
    return jsonify(models=[i[:-7] for i in models.get_models_list()]), 200


@app.route('/create_model', methods=["GET"])
def create_model():
    """
    Function creates new model with specified name,class and hyperparameters from request
    :return: message that model was created
    """
    name = request.args.get("name")
    class_name = request.args.get("class")
    params = request.args.get("params")
    if params:
        params = json.loads(params)
    try:
        name = models.create(class_name, name, params)
        return f"Model {name} has been created", 200
    except KeyError as error:
        return str(error), 500


@app.route('/delete', methods=['GET'])
def delete():
    """
    Function deletes model with the given name
    :return: message that model has been deleted
    """
    name = request.args.get("name")
    models.delete_model(name)
    return f"Model {name} has been deleted", 200


@app.route('/train', methods=['GET'])
def train():
    """
    Function gets name of the model and training data from request and trains the model
    :return: Message that training was successful
    """
    data = request.get_json()
    name = request.args.get("name")
    data = pd.DataFrame.from_dict(data)
    try:
        models.train(name, data)
    except AttributeError as error:
        return str(error), 500
    except KeyError as error:
        return str(error), 500
    return "Training successful", 200


@app.route('/predict', methods=['GET'])
def predict():
    """
    Function gets data and model name from request,
    then loads model and uses it for predictions
    :return: predictions of the model in float type
    """
    data = request.get_json()
    model_name = request.args.get("name")
    data = pd.DataFrame.from_dict(data)
    model = models.get(model_name)
    try:
        predictions = model.predict(data)
    except AttributeError:
        return str(AttributeError("Model with the given name doesn't exist")), 500
    return jsonify(Predictions=list(map(float, predictions))), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0')
