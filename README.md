# Consistency Trajectory Models (CTM) - Training & Sampling Guide
 
A simple Jupyter notebook implementation for learning and experimenting with Consistency Trajectory Models using the CIFAR-10 dataset.
 
## Overview
 
This project provides a Docker-containerized Jupyter notebook for implementing Consistency Trajectory Models (CTM), specifically designed for training from scratch and sample generation.

## Prerequisites

* Docker installed on your system
* NVIDIA GPU with CUDA support
* Required files:
  - `model.pt` (pre-trained model checkpoint)
  - `cifar10-32x32.npz`

## Setup Instructions

### 1. Build Docker Image

```bash
# Build the Docker image for CTM
docker build -t ctm-environment .
```

### 2. Create and Launch Docker Container

```bash
# Create and run the container with GPU support
docker run -it --name=ctm-container --gpus=all --ipc=host --network=host -d -v /path-to-mount/:/workspace ctm-environment
```

```bash
# Launch the container
docker exec -it ctm-container bash
```

### 3. Prepare Files Inside Container

The model checkpoint file and cifar10-32x32.npz file can be downloaded from the following link - [model](https://drive.google.com/drive/folders/1mi_HN7a7k6aEm8wl0ooMlRwccY-31-d_) & [cifar](https://drive.google.com/drive/folders/1ei4PLmTrAlj-j_yUfLqXSpI5OOIqDlgv)

Once inside the container:

```bash
# Move model checkpoint to appropriate location
mv model095000.pt /mount-location/

# Move CIFAR-10 reference dataset to appropriate location
mv cifar10-32x32.npz /mount-location/
```

### 4. Launch Jupyter Notebook

```bash
# Start Jupyter notebook server
jupyter notebook --allow-root --no-browser --port=8080 --ip 0.0.0.0
```

Now open the ctm-notebook.ipynb in jupyter environment

## Training Configuration

### Hyperparameters

Modify the hyperparameters in the notebook if required according to the setup.

### Key Training Parameters

* **Batch Size**: 128 (with microbatch size of 8)
* **EMA Start**: 0.9999
* **Save Interval**: Every 10,000 steps
* **Evaluation Interval**: Every 1,000 steps
* **ref_path**: Approriate path
* **out_dir**: Appropriate path
* **resume_checkpoint**: Appropriate path

## Sampling Configuration

After training completes, configure sampling hyperparameters similarly if required:

### Sampling Hyperparameters

### Key Sampling Parameters

* **Model Path**: Path to trained checkpoint (target_modelxxxxxx.pt)
* **Sampling Steps**: 100
* **Batch Size**: 8
* **Sampler Type**: Exact sampler
* **Class Conditional**: False (unconditional generation)
* **Number of Samples**: 8

## Project Structure

```
.
├── Dockerfile
├── model095000.pt             # Pre-trained model checkpoint
├── cifar10-32x32.npz          # Reference dataset
├── cifar10_images/            # Training data directory
├── notebooks/                 # Jupyter notebooks
└── output/                    # Generated samples and checkpoints
    └── GAN/
        └── uncond/
            └── GAN_bs_528_ema_0.9999_diff_aug/
```

## Usage Workflow

1. **Setup Environment**: Build Docker image and create container
2. **Launch Notebook**: Start Jupyter notebook server
3. **Prepare Data**: Will be downloaded once the dataset download cell is executed
4. **Training**: Set training hyperparameters and run training cells
5. **Monitoring**: Check FID scores and similarity metrics during training
6. **Sampling**: Configure sampling parameters and generate images
7. **Evaluation**: Analyze generated samples and metrics


## Troubleshooting

* **Out of Memory**: Reduce `batch_size` or `microbatch` size
* **Path Errors**: Verify all file paths match your directory structure
* **GPU Issues**: Check CUDA installation and GPU availability

---

## Contributors
* [Harshavardhan R](https://www.linkedin.com/in/harsha-tesla/) 
* [Srinidhi Srinivasa](https://www.linkedin.com/in/gonchkarsrinidhi/)
* [Basavaraj Murali](https://www.linkedin.com/in/basavaraj-murali-022819121/)
* [Ritesh Mahajan](https://www.linkedin.com/in/ritesh-mahajan-2b318250/)
