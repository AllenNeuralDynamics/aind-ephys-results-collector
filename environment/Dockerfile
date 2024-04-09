# hash:sha256:0ecef861c6d975f5c81da8cbe1a36a521c12aa551295605cbd16152ddf184694
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.0.9-miniconda4.9.2-python3.8-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    aind-data-schema==0.33.5 \
    aind-metadata-upgrader==0.0.6 \
    spikeinterface[full]==0.100.1
