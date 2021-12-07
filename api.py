from flask import Flask, request, jsonify
from models import Models
import json

import pandas as pd

app = Flask(__name__)


models = Models()


@app.route('/')
def hello_world():
    return 'This is ML deploy app'


@app.route('/classes', methods=["GET"])
def get_classes():
    """

    :return:
    """
    return jsonify(classes=list(models.classes.keys())), 200


@app.route('/models', methods=["GET"])
def get_models():
    """
    Используя функцию get_models_list класса Models полуаем список моделей
    i[:-7] удаляет '.pickle' из имен моделей
    :return: json со списком доступных моделей
    """
    return jsonify(models=[i[:-7] for i in models.get_models_list()]), 200


@app.route('/create_model', methods=["GET"])
def create_model():
    """

    :return: сообщение о том, что модель была создана
    """
    name = request.args.get("name")
    class_name = request.args.get("class")
    params = request.args.get("params")
    params = json.loads(params)
    name = models.create(class_name, name, params)
    return f"Model {name} has been created", 200


@app.route('/delete', methods=['GET'])
def delete():
    """

    :return:
    """
    name = request.args.get("name")
    models.delete_model(name)
    return f"Model {name} has been deleted", 200


@app.route('/train', methods=['POST'])
def train():
    """


    :return: Message that training has benn successful
    """
    data = request.get_json()
    name = request.args.get("name")
    data = pd.DataFrame.from_dict(data)
    models.train(name, data)
    return "Training successful", 200


@app.route('/predict', methods=['GET'])
def predict():
    """

    :return:
    """
    data = request.get_json()
    model_name = request.args.get("name")
    data = pd.DataFrame.from_dict(data)
    model = models.get(model_name)
    predictions = model.predict(data)
    return jsonify(Predictions=list(map(float, predictions))), 200


if __name__ == '__main__':
    app.run()
