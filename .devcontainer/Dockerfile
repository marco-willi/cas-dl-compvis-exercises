FROM pytorch/pytorch:2.14.1-cuda12.6-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl wget ca-certificates git \
    build-essential make \
    jq unzip less \
    ffmpeg libglib2.0-0 libsm6 libxext6 libxrender1 libgl1 \
    graphviz \
    bash bash-completion zsh \
    shellcheck \
    && rm -rf /var/lib/apt/lists/*
