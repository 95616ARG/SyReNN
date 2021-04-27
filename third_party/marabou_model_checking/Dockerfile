FROM ubuntu:18.04

LABEL maintainer="masotoud@ucdavis.edu"
LABEL autodelete="True"

RUN apt-get update && apt-get install -y \
    wget unzip build-essential

# Install benchexec.
RUN wget https://github.com/sosy-lab/benchexec/releases/download/2.0/benchexec_2.0-1_all.deb -O benchexec.deb
RUN apt install -y --install-recommends ./benchexec.deb

# Get the Marabou source.
WORKDIR /
RUN wget https://github.com/NeuralNetworkVerification/Marabou/archive/0a6b4638f0df700876d5a04e7e6661a34a649231.zip -O marabou.zip
RUN unzip marabou.zip -d marabou
RUN rm marabou.zip
RUN mv marabou/**/* marabou
RUN rm -r marabou/Marabou-*

# Build Marabou.
WORKDIR /marabou
RUN make

# Build our executable.
WORKDIR /marabou/cpp_interface_example
COPY Makefile Makefile
COPY bmc.cpp bmc.cpp
RUN make

RUN rm -rf /var/lib/apt/lists/*
