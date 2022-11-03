FROM python:3.9-buster

WORKDIR /app
RUN mkdir -p /results/last-inference/
RUN mkdir -p /results/last-evaluation/

COPY . .

RUN pip3 install .
