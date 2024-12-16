# hash:sha256:12d6c62a77e670d62220f89ff3b046615c6067863c54055e1fe13b48b6026897
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    aind-data-schema==1.0.0 \
    aind-log-utils==0.2.1 \
    aind-metadata-upgrader==0.0.18 \
    spikeinterface[full]==0.101.2
