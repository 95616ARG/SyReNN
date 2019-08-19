FROM caffett/vrl_env

LABEL maintainer="masotoud@ucdavis.edu"
LABEL autodelete="True"

RUN apt-get update && apt-get install -y \
    wget unzip build-essential git

# Get the VRL source
WORKDIR /
RUN git clone https://github.com/caffett/VRL_CodeReview.git vrl

WORKDIR /vrl
RUN git checkout 65598586e3338b1d3ebea69b4d037a9c177de1cb

COPY Makefile.docker Makefile
