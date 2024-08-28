# hash:sha256:37d413803c07d1b2ad038b9e85672c0de3ca8f63ccc3e1f4322df15d798b1c1e
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
