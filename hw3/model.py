# 以Sequential序列模型進行LSTM模型建構並訓練，預測CNC加工數據的任務

import torch.nn as nn

class CncPredictor(nn.Module):
    def __init__(self, input_size=45, hidden_size=30, output_size=45, num_layers=2, dropout=0.2):
        super(CncPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.model = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size//2, output_size)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.model(lstm_out[:, -1, :])
        return out