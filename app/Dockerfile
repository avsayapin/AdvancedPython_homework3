FROM python:3.9

LABEL MAINTANER = "Artyom Sayapin <tema3544@yandex.ru>"

ENV CELERY_BROKER redis://redis:6379/0
ENV CELERY_BACKEND redis://redis:6379/0

COPY . /app
WORKDIR /app

RUN  pip install -r requirements.txt

EXPOSE 5000
ENTRYPOINT ["python3"]
CMD ["api.py"]