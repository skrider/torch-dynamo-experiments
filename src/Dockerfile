FROM nvidia/cuda:12.2.0-devel-ubuntu20.04 AS base

RUN apt-get update -q \
    && apt-get install -q -y --no-install-recommends \
    bzip2 ca-certificates git openssh-client procps wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* # buildkit

ENV CONDA_VERSION=py311_23.5.2-0
ENV MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-${CONDA_VERSION}-Linux-x86_64.sh"

RUN wget "${MINICONDA_URL}" -O miniconda.sh -q \
    && mkdir -p /opt \
    && bash miniconda.sh -b -p /opt/conda \
    && rm miniconda.sh \
    && ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh \
    && echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc \
    && echo "conda activate base" >> ~/.bashrc \
    && find /opt/conda/ -follow -type f -name '*.a' -delete \
    && find /opt/conda/ -follow -type f -name '*.js.map' -delete \
    && /opt/conda/bin/conda clean -afy

RUN --mount=type=cache,target=/root/.conda/pkgs \
    --mount=type=cache,target=/opt/conda/pkgs \
    /opt/conda/bin/conda install --solver libmamba \
    pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch-nightly -c nvidia

RUN --mount=type=cache,target=/root/.cache/pip \
    /opt/conda/bin/conda run python -m pip install \
    gpustat transformers tabulate tensorboard tensorboardX torch-tb-profiler

RUN git clone https://github.com/pytorch/benchmark
COPY models.txt /models.txt
RUN /opt/conda/bin/conda run python -u /benchmark/install.py $(cat models.txt | sed '/^# /d')
RUN /opt/conda/bin/conda run python -m pip install /benchmark

FROM base AS final

COPY . /src
# for some reason have to install editable
RUN /opt/conda/bin/conda run python -m pip install -e /src

# for some reason torchbenchmark needs to be imported in its root directory
WORKDIR /benchmark

ENTRYPOINT ["/opt/conda/bin/conda", "run", "python", "-m", "torch_dynamo_experiments.cmd.run"]

