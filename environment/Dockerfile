# hash:sha256:94c0d4f347bbb5c1e4a2ec7fe9950bae01239555c60ad19c87d4b2a1519470e2
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    aind-data-schema==1.0.0 \
    aind-metadata-upgrader==0.0.17 \
    spikeinterface[full]==0.101.1
