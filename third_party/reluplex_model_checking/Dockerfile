FROM ubuntu:18.04

LABEL maintainer="masotoud@ucdavis.edu"
LABEL autodelete="True"

RUN apt-get update && apt-get install -y \
    wget unzip build-essential

# Install benchexec.
RUN wget https://github.com/sosy-lab/benchexec/releases/download/2.0/benchexec_2.0-1_all.deb -O benchexec.deb
RUN apt install -y --install-recommends ./benchexec.deb

# Get the Reluplex source.
WORKDIR /
RUN wget https://github.com/guykatzz/ReluplexCav2017/archive/e2e48b2980f76bdff74787427f43fff3b26a45a4.zip -O reluplex.zip
RUN unzip reluplex.zip -d reluplex
RUN rm reluplex.zip
RUN mv reluplex/**/* reluplex
RUN rm -r reluplex/ReluplexCav2017-*

# Install Reluplex dependency (modified GLPK).
WORKDIR /reluplex/glpk-4.60
RUN ./configure_glpk.sh
RUN make
RUN make install

# Build ReluPlex.
WORKDIR /reluplex/reluplex
RUN make

# Build our executable.
WORKDIR /reluplex
COPY Makefile Makefile
COPY bmc.cpp bmc.cpp
RUN make

RUN rm -rf /var/lib/apt/lists/*
