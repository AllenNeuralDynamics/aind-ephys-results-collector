# hash:sha256:517f02b49788c65fd6fd3a26b456b5eeacfb71542a8b9a64a061f3139d46d486
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:24.5.0-0-python3.12.4-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-ask-pass /

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential=12.9ubuntu3 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -U --no-cache-dir \
    aind-data-schema==2.5.1 \
    aind-log-utils==0.2.3 \
    aind-metadata-upgrader==0.14.4 \
    spikeinterface[full]==0.103.0
