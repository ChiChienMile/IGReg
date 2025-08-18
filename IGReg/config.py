import torch

# Check if GPU is available; use GPU if possible, otherwise use CPU
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Number of threads for data loading (DataLoader num_workers)
n_threads = 1

# Number of samples per batch
batch_size = 1

# Random seed for reproducibility
random_seed = 1337

# Initial learning rate
lr_self = 1e-4

# Minimum learning rate (lower bound during LR decay)
min_lr = 1e-6

# Weight decay coefficient (L2 regularization)
weight_decay = 2e-5

# Total number of training epochs
epochs = 100

# Starting epoch for saving model checkpoints (epochs < save_epoch will not be saved)
save_epoch = 0

# Dataset root directory (change to your actual dataset path)
data_dir = './'