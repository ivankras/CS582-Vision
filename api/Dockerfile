#FROM python:3.8
FROM nvidia/cuda:10.2-base

RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y curl apt-utils && apt-get install -y gfortran libopenblas-dev liblapack-dev
RUN apt-get install unzip
RUN apt-get -y install python3.8
RUN apt-get -y install python3-pip

WORKDIR /usr/src/app

COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY . ./api



EXPOSE 5000


CMD [ "flask", "run", "--host", "0.0.0.0" ]
