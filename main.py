from Dataset import TaskDataset, get_transform, convert_to_rgb
from config import BATCH_SIZE, device
from RobustEncoder import RobustEncoderWithClassifier
from train import train

import torch
from torch.utils.data import Dataset, DataLoader, random_split

dataset: TaskDataset = torch.load("./dataset/Train.pt", map_location=device, weights_only=False)
dataset.transform = get_transform()
print(f"Loaded dataset with {len(dataset)} samples.")

# [8.5]
# Set split sizes
train_size = int(0.9 * len(dataset))
test_size = len(dataset) - train_size

# Perform split
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f" Train size: {len(train_dataset)} | Test size: {len(test_dataset)}")

# [9.0]
model = RobustEncoderWithClassifier().to(device)
losses = train(model, train_loader, test_loader)