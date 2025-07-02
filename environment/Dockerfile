# hash:sha256:1d2e030f4ee3804a2eed20c8e96974fe5283187b6e2df0b052c0538cb01c192c
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    aind-data-schema==1.3.0 \
    aind-log-utils==0.2.3 \
    aind-metadata-upgrader==0.0.24 \
    spikeinterface[full]==0.102.3

COPY postInstall /
RUN /postInstall
