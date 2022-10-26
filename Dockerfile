FROM python:3.7-slim-buster

WORKDIR /app

RUN yes Y | apt-get update \
    && yes Y | apt-get upgrade \
    && yes Y | apt-get install git \
    && yes Y | apt-get install wget \
    && yes Y | apt-get install curl \
    && yes Y | apt-get install -y libpq-dev \
    && yes Y | apt install python3-dev python3-pip \
    && pip3 install --upgrade pip \
    && curl -sSL https://sdk.cloud.google.com | bash

COPY requirements.txt /app

RUN pip3 --no-cache-dir install -r requirements.txt

COPY . /app

#ENV PATH $PATH:/root/google-cloud-sdk/bin

#RUN gcloud auth activate-service-account --key-file=bq_creds.json

#ENV GOOGLE_APPLICATION_CREDENTIALS=bq_creds.json

EXPOSE 5000

CMD ["python3", "app.py"]
