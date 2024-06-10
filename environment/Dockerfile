# hash:sha256:b9a3abafa86ef3f652a13d0141fd7d4d12d6c330d1547366a3a66b85bc1f520d
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.0.9-miniconda4.9.2-python3.8-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    aind-data-schema==0.38.0 \
    aind-metadata-upgrader==0.0.8 \
    spikeinterface[full]==0.100.7
