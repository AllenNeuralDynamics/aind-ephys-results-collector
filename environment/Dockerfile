# hash:sha256:69c72e1fad1e1491a4af46931061107337102657ab64dc99611f5dc348754596
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    aind-data-schema==1.0.0 \
    aind-metadata-upgrader==0.0.18 \
    spikeinterface[full]==0.101.1

COPY postInstall /
RUN /postInstall
