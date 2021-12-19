FROM python:3.9

ENV MLFLOW_TRACKING_URI http://mlflow:5050

RUN pip install mlflow

ADD . /mlflow
WORKDIR /mlflow

EXPOSE 5050