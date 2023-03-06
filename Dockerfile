FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

COPY requirements.txt requirements.txt

ARG DEBIAN_FRONTEND=noninteractive
ARG PIP_PREFER_BINARY=1
ARG PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends \
  git \
  tesseract-ocr \
  && pip3 install -r requirements.txt \
  && pip3 install autokeras
