# 以Sequential序列模型進行LSTM模型建構並訓練，預測CNC加工數據的任務

import torch
import torch.nn as nn

class CncPredictor(nn.Module):
    def __init__(self, input_size=45, hidden_size=30, output_size=45, num_layers=2, dropout=0.2):
        super(CncPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # Take the output from the last time step
        last_time_step = lstm_out[:, -1, :]
        out = self.linear(last_time_step)
        return out