FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    ca-certificates \
    clang \
    cmake \
    curl \
    g++ \
    gcc \
    git \
    jq \
    make \
    openjdk-17-jdk \
    pkg-config \
    python3 \
    python3-pip \
    python3-venv \
    unzip \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /workspace
COPY . /workspace
RUN python3 -m pip install --upgrade pip setuptools wheel \
    && python3 -m pip install -e . -r requirements.txt

CMD ["bash"]
