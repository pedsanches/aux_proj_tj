FROM python:3.10-slim

WORKDIR /inference

COPY transcription_models /inference/transcription_models

COPY sentence_classification /inference/sentence_classification

COPY requirements.txt requirements.txt
COPY main.py main.py

RUN pip install -r requirements.txt

EXPOSE 8011
ENV GUNICORN_CMD_ARGS="--bind=0.0.0.0:8011 -k uvicorn.workers.UvicornWorker --timeout 1000"

ENTRYPOINT exec gunicorn main:app