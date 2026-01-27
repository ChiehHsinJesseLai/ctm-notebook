FROM dongjun57/ctm-docker:latest

RUN apt update && apt -y upgrade && \
    apt -y install curl git vim tmux wget software-properties-common \
    python3 python3-dev python3-pip python3-setuptools python3-wheel \
    libopenmpi-dev

RUN pip install tensorflow && \
    pip install torch torchvision torchaudio && \
    pip install blobfile tqdm numpy scipy pandas Cython piq==0.7.0 && \
    pip install joblib==0.14.0 albumentations==0.4.3 lmdb pillow && \
    pip install clip@git+https://github.com/openai/CLIP.git && \
    pip install flash-attn --no-build-isolation && \
    pip install xformers && \
    pip install mpi4py && \
    pip install nvidia-ml-py3 timm==0.4.12 legacy dill

# Install Jupyter Notebook and JupyterLab
RUN pip install jupyter jupyterlab notebook
RUN pip install gdown
RUN pip install matplotlib

