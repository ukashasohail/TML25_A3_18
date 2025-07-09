import torch
import random

# Set seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# Hyperparameters
EPSILON = 0.0314
ALPHA = 0.007
MAX_ITERS = 7
LEARNING_RATE = 0.001
BATCH_SIZE = 64
EPOCHS = 25
TEMPERATURE = 0.5
WEIGHT = 2 / 256

clamp_min = 0.0
clamp_max = 1.0