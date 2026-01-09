# kohya-docker/Dockerfile
FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs \
    ca-certificates \
    python3-dev \
    build-essential \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# sd-scripts
WORKDIR /opt
RUN git clone --depth 1 https://github.com/kohya-ss/sd-scripts.git

WORKDIR /opt/sd-scripts

# sd-scripts requirements (torch is NOT included by design)
# (必要に応じてバージョン固定は運用で)
RUN pip install --upgrade pip && \
    pip install "numpy<2" && \
    pip install -r requirements.txt

# accelerate は train_network.py を叩くときに使うことが多いので入れておく
RUN pip install --upgrade accelerate

# デフォルト作業ディレクトリ（ホストを /workspace にマウントして使う）
WORKDIR /workspace
CMD ["bash"]
