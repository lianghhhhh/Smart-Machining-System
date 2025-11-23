# 採用任何一種CNN模型，完成鋼鐵產品表面的多元瑕疵檢測

from torch import nn

class CnnModel(nn.Module):
    def __init__(self):
        super(CnnModel, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1), # 32, 1600, 256
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(4, 4), # 32, 400, 64

            nn.Conv2d(32, 64, 3, 1, 1), # 64, 400, 64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(4, 4), # 64, 100, 16

            nn.Conv2d(64, 128, 3, 1, 1), # 128, 100, 16
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 128, 50, 8

            nn.Conv2d(128, 256, 3, 1, 1), # 256, 50, 8
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 256, 25, 4

            nn.Flatten(),
            nn.Linear(256 * 25 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 5) # 0-4 classfication
        )
        
    def forward(self, x):
        return self.model(x)