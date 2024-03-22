# hash:sha256:bac3d133f8cb44a72290a18c5607c36d9ecf571aa926cc3a46d639f42ca68465
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.0.9-miniconda4.9.2-python3.8-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    aind-data-schema==0.31.8 \
    aind-metadata-upgrader==0.0.6 \
    spikeinterface[full]==0.100.1
