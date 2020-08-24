FROM ubuntu:18.04

LABEL maintainer="masotoud@ucdavis.edu"
LABEL autodelete="True"

RUN apt-get update && apt-get install -y \
    wget unzip build-essential python3 \
    python3-distutils python3-pip git gdb

# Get the ERAN source
WORKDIR /
RUN wget https://github.com/eth-sri/eran/archive/31f7784bea1cf00ab2e358b12e68414893122034.zip -O eran.zip
RUN unzip eran.zip -d eran && rm eran.zip
RUN mv eran/**/* eran && rm -r eran/eran-*

WORKDIR /eran
# We use our own install.sh to specify the ELINA version (for reproducibility)
# and use multi-threaded Make.
COPY install.sh install.sh
RUN ./install.sh
RUN pip3 install numpy tensorflow matplotlib tqdm Pillow

# Install benchexec
RUN wget https://github.com/sosy-lab/benchexec/releases/download/2.0/benchexec_2.0-1_all.deb -O benchexec.deb
RUN apt install -y --install-recommends ./benchexec.deb

RUN rm -rf /var/lib/apt/lists/*
