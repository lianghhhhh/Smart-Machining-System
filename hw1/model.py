# 以Sequential序列模型進行CNN模型建構並訓練，完成太陽能板表面的二元瑕疵分類的測試

from torch import nn

class CnnModel(nn.Module):
    def __init__(self):
        super(CnnModel, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1), # 32, 300, 300
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(3, 3), # 32, 100, 100

            nn.Conv2d(32, 64, 3, 1, 1), # 64, 100, 100
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 64, 50, 50

            nn.Conv2d(64, 128, 3, 1, 1), # 128, 50, 50
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 128, 25, 25

            nn.Flatten(),
            nn.Linear(128 * 25 * 25, 128),
            nn.ReLU(),
            nn.Linear(128, 2) # 0 or 1 classification
        )
        
    def forward(self, x):
        return self.model(x)