import torch as th
import numpy as np
import random

def set_seed(seed):
    # Set seed for NumPy
    np.random.seed(seed)
    # Set seed for Python's random module
    random.seed(seed)
    # Set seed for PyTorch
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    
    # Ensure reproducibility in cuDNN
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False
