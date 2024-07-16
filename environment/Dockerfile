# hash:sha256:88094fe580405ed4ab3e3154c1098920fb03c52809fc6d1561bac3718fdbecc5
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    aind-data-schema==0.38.0 \
    aind-metadata-upgrader==0.0.9

COPY postInstall /
RUN /postInstall
