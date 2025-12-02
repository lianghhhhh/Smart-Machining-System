# 採用任何一種CNN模型，完成鋼鐵產品表面的多元瑕疵檢測

from torch import nn
from torchvision import models

class BinaryModel(nn.Module):
    def __init__(self):
        super(BinaryModel, self).__init__()
        self.model = models.resnet18(weights="IMAGENET1K_V1")
        self.model.conv1 = nn.Conv2d(1, 64, 7, 2, 3)
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 2)
        )
        
    def forward(self, x):
        return self.model(x)
    

class MultiClassModel(nn.Module):
    def __init__(self):
        super(MultiClassModel, self).__init__()
        self.model = models.efficientnet_b0(weights="IMAGENET1K_V1")
        # Change first conv layer to accept 1 channel
        self.model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.model.classifier[1] = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.model.classifier[1].in_features, 4)
        )
        
    def forward(self, x):
        return self.model(x)