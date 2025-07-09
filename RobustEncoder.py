import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models

# [3.0]
class RobustEncoderWithClassifier(nn.Module):
    def __init__(self, proj_dim=128, num_classes=10):
        super().__init__()
        base = models.resnet18(weights=None)
        self.encoder = nn.Sequential(*list(base.children())[:-1], nn.Flatten())
        self.projector = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, proj_dim)
        )
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        feat = self.encoder(x)
        z = F.normalize(self.projector(feat), 1)
        logits = self.classifier(feat)
        return z, logits
