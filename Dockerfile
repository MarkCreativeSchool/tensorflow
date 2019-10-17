FROM continuumio/anaconda3:2019.07

RUN . /opt/conda/bin/activate \
  && conda install -y cudatoolkit

ADD . /app
WORKDIR /app

RUN . /opt/conda/bin/activate \
  && pip install -r requirements.txt