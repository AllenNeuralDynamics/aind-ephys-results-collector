# hash:sha256:0d81f5a31677e1421d48d87a0ef5046416d325a5c20a5f61da67911615c77e05
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    aind-data-schema==1.3.0 \
    aind-log-utils==0.2.3 \
    aind-metadata-upgrader==0.0.24 \
    spikeinterface[full]==0.103.0
