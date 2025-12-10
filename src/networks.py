import torch.nn as nn
from torchvision import models

class AgeEstimator(nn.Module):
    def __init__(self, pretrained=True):
        super(AgeEstimator, self).__init__()
        
        # 1. Load EfficientNet-B0
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.backbone = models.efficientnet_b0(weights=weights)
        
        # 2. Modify the Classifier
        # EfficientNet-B0's last layer is named 'classifier'
        # The input features for B0 is 1280 (vs 512 for ResNet18)
        in_features = self.backbone.classifier[1].in_features
        
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        return self.backbone(x)