# hash:sha256:8ae7a1c3c27b9ce0109c96a7bcaf7a238750cd6d7ce2db292195491351ae8096
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    aind-data-schema==0.38.5 \
    aind-metadata-upgrader==0.0.9 \
    spikeinterface[full]==0.101.0
