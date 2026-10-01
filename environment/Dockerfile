# hash:sha256:ca3d6bfc569af87c380d57957ca26fa03bd5531a13e54440b75edd41d6fe39c6
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:24.5.0-0-python3.12.4-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-ask-pass /

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential=12.9ubuntu3 \
    && rm -rf /var/lib/apt/lists/*

RUN conda install -y \
        jupyterlab==4.4.6 \
    && conda clean -ya

RUN pip install -U --no-cache-dir \
    aind-data-schema==2.8.1 \
    aind-metadata-upgrader==0.17.1 \
    log-schema==0.2.10.dev1 \
    spikeinterface[full]==0.105.0
