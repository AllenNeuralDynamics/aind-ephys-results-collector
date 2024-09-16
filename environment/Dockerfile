# hash:sha256:e31edb7c9e49423be5aea30f41e4987aadcad9e1f4840b44b72989ef78d5646b
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    aind-data-schema==1.0.0 \
    aind-metadata-upgrader==0.0.14 \
    spikeinterface[full]==0.101.0

COPY postInstall /
RUN /postInstall
