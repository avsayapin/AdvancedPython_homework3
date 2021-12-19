FROM python:3.9

LABEL MAINTANER = "Artyom Sayapin <tema3544@yandex.ru>"

ENV CELERY_BROKER redis://redis:6379/0
ENV CELERY_BACKEND redis://redis:6379/0
ENV MLFLOW_TRACKING_URI http://mlflow:5050


COPY . /worker
WORKDIR /worker

RUN  pip install -r requirements.txt

EXPOSE 5000
ENTRYPOINT celery -A tasks worker --loglevel=info --pool=solo