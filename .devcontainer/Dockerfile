ARG BASE_IMAGE
FROM $BASE_IMAGE as base

ENV SHELL /bin/bash
SHELL [ "/bin/bash", "-c" ]

ENV DEBIAN_FRONTEND=noninteractive

RUN echo ${DEBIAN_FRONTEND}

RUN pip3 install \
    gym \
    tensorflow \
    Pillow \
    matplotlib \
    jupyter \
    pytest \
    docopt \
    pyyaml \
    protobuf \
    grpcio \
    pandas \
    scipy \
    ipykernel

