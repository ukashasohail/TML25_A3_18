import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

def convert_to_rgb(img):
    return img.convert("RGB") if img.mode != "RGB" else img

def get_transform():
    color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
    random_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    random_gray = transforms.RandomGrayscale(p=0.2)
    return transforms.Compose([
        random_color_jitter,
        random_gray,
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(32),
        transforms.Lambda(convert_to_rgb),
        transforms.ToTensor(),
    ])

class TaskDataset(Dataset):
    def __init__(self, ids=None, imgs=None, labels=None, transform=None):
        self.ids = ids if ids is not None else []
        self.imgs = imgs if imgs is not None else []
        self.labels = labels if labels is not None else []
        self.transform = get_transform()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        id_ = self.ids[index]
        original_image = self.imgs[index]
        view1 = self.transform(original_image)
        view2 = self.transform(original_image)
        label = self.labels[index]
        original_tensor = transforms.ToTensor()(convert_to_rgb(original_image))
        return original_tensor, view1, view2, id_, label