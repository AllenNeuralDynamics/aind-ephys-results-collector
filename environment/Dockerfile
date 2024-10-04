# hash:sha256:b8c0198139861f10a0c64ca91cb3ba65c9f74200ac82e79c34a034950a4907c2
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    aind-data-schema==1.0.0 \
    aind-metadata-upgrader==0.0.18 \
    spikeinterface[full]==0.101.2
