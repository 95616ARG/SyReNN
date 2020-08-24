FROM ubuntu:18.04

LABEL maintainer="masotoud@ucdavis.edu"
LABEL autodelete="True"

# https://stackoverflow.com/questions/24288616#answer-48536224
ARG UID=1000

# Bazel, build dependencies, benchexec dependencies, and docker-in-docker
# dependencies.
RUN apt-get update && apt-get install -y \
    pkg-config zip g++ zlib1g-dev unzip \
    python2.7 python-pip python3 python3-pip \
    build-essential curl git cmake libcairo2 libffi-dev libgmp3-dev \
    zlib1g-dev libssl-dev libbz2-dev zip \
    apt-transport-https ca-certificates gnupg-agent software-properties-common

# Install Docker.
RUN curl -fsSL https://download.docker.com/linux/ubuntu/gpg | apt-key add -
RUN add-apt-repository \
        "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
        $(lsb_release -cs) \
        stable"
RUN apt-get update && apt-get install -y docker-ce docker-ce-cli containerd.io

# Install Bazel.
WORKDIR /
RUN curl -OL https://github.com/bazelbuild/bazel/releases/download/1.1.0/bazel-1.1.0-installer-linux-x86_64.sh && \
    chmod +x bazel-1.1.0-installer-linux-x86_64.sh && \
    ./bazel-1.1.0-installer-linux-x86_64.sh && \
    rm bazel-1.1.0-installer-linux-x86_64.sh

# Set up bazel_python
RUN git clone https://github.com/95616ARG/bazel_python.git && \
    cd bazel_python && \
    echo y | ./setup_python.sh 3.7.4 /.bazel_python && \
    cp $HOME/.bazelrc /etc/bazel.bazelrc && \
    cd .. && \
    rm -rf bazel_python

# Install benchexec.
RUN curl -OL https://github.com/sosy-lab/benchexec/releases/download/2.2/benchexec_2.2-1_all.deb && \
    apt install -y --install-recommends ./benchexec_2.2-1_all.deb && \
    rm benchexec_2.2-1_all.deb

RUN rm -rf /var/lib/apt/lists/*

# Store Bazel outputs on /iovol/.docker_bazel, which will be loaded at runtime.
# This allows us to cache Bazel build artifacts across Docker invocations.
RUN echo "startup --output_user_root=/iovol/.docker_bazel" >> /etc/bazel.bazelrc

RUN useradd -u $UID -ms /bin/bash docker_user
RUN adduser docker_user benchexec
RUN adduser docker_user docker

EXPOSE 50051
USER docker_user
