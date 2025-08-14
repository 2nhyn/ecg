FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

## DO NOT EDIT these 3 lines.
RUN mkdir /challenge
COPY ./ /challenge
WORKDIR /challenge


## Install your dependencies here using apt install, etc.
RUN apt update && apt install -y \
    build-essential \
    libatlas-base-dev \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6

## Include the following line if you have a requirements.txt file.
RUN pip install -r requirements.txt