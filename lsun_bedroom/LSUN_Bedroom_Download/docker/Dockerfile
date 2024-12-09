FROM nvcr.io/nvidia/pytorch:23.07-py3
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends git

# conda env setup
# RUN conda init bash
# ARG conda_env=cbdm
# RUN conda create --name $conda_env python=3.8 -y
# ENV PATH /opt/conda/envs/$conda_env/bin:$PATH
# ENV CONDA_DEFAULT_ENV $conda_env

# RUN /bin/bash -c "source activate $conda_env"

# required installations
COPY ./requirements.txt .
RUN pip install -r requirements.txt

# for fixing issues with opencv circular import.
# Opencv dependencies
# RUN apt-get update && apt-get -y install libopencv-dev python3-opencv
RUN pip uninstall -y opencv-python && pip install opencv-python==4.5.5.64

# setting few path variables for pytorch
ENV PATH="${PATH}:/opt/hpcx/ompi/bin"
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/opt/hpcx/ompi/lib"

