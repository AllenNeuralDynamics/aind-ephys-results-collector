# hash:sha256:26eb46b6754f36395c645709f51d11374070199130575608895a3d6465cbbbb9
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    aind-data-schema==0.31.8 \
    aind-metadata-upgrader==0.0.6

COPY postInstall /
RUN /postInstall
