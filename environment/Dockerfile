# hash:sha256:4ec7f196ddef5e7c9b0c38ca7c5bb6674d5ae4dc5b306aa65e112bf969ed7645
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    aind-data-schema==1.0.0 \
    aind-log-utils==0.2.3 \
    aind-metadata-upgrader==0.0.18 \
    spikeinterface[full]==0.101.2
