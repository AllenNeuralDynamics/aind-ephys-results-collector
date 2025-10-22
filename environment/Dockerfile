# hash:sha256:c50800870cee19fb2c3cdc5da98bd625ede95743db2c64c253151d952ba3a290
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:24.5.0-0-python3.12.4-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-askpass /

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential=12.9ubuntu3 \
    && rm -rf /var/lib/apt/lists/*

RUN conda install -y \
        jupyterlab==4.4.6 \
    && conda clean -ya

RUN pip install -U --no-cache-dir \
    aind-data-schema==2.0.1 \
    aind-log-utils==0.2.3 \
    aind-metadata-upgrader==0.2.2 \
    spikeinterface[full]==0.103.0
