FROM dongjun57/ctm-docker:latest
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-dev libopenmpi-dev git wget \
    python3 python3-dev python3-pip python3-setuptools python3-wheel \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get install libopenmpi-dev -y
# conda env setup
# RUN conda init bash
# ARG conda_env=ctm
# RUN conda create --name $conda_env python=3.8 -y
# ENV PATH /opt/conda/envs/$conda_env/bin:$PATH
# ENV CONDA_DEFAULT_ENV $conda_env
# RUN /bin/bash -c "source activate $conda_env"

# required installations
COPY ./requirements.txt .
RUN pip install -r requirements.txt

RUN pip install tensorflow[and-cuda]
RUN pip install torch torchvision torchaudio
RUN pip install blobfile tqdm numpy scipy pandas Cython piq==0.7.0
RUN pip install joblib==0.14.0 albumentations==0.4.3 lmdb clip@git+https://github.com/openai/CLIP.git pillow
RUN pip install flash-attn --no-build-isolation
RUN pip install mpi4py
RUN pip install nvidia-ml-py3 timm==0.4.12 legacy dill nvidia-ml-py3

# for fixing issues with opencv circular import.
# Opencv dependencies
# RUN apt-get update && apt-get -y install libopencv-dev python3-opencv
RUN pip uninstall -y opencv-python && pip install opencv-python==4.8.0.76

# setting few path variables for pytorch
ENV PATH="${PATH}:/opt/hpcx/ompi/bin"
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/opt/hpcx/ompi/lib"

RUN ln -s /usr/bin/python3 /usr/bin/python

