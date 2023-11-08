# hash:sha256:2cd9649fc74dad892daa2a4972e78733be3238220c40a2d4c61fd6856d82bc1f
FROM registry.codeocean.allenneuraldynamics-test.org/codeocean/jupyterlab:3.0.9-miniconda4.9.2-python3.8-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    aind-data-schema==0.15.25
