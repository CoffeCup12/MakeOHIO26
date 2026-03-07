import torch
import torch.nn as nn
import torch.nn.functional as F

class LineRatePredictor(nn.Module):
    def __init__(self, num_features):
        super(LineRatePredictor, self).__init__()

        self.conv1 = nn.Conv1d(num_features, 32, kernel_size=3)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3)

        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=32,
            batch_first=True
        )

        self.fc1 = nn.Linear(32, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        # x shape: (batch, time, features)

        x = x.permute(0, 2, 1)       # → (batch, features, time)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x = x.permute(0, 2, 1)       # → (batch, time, channels)

        x, _ = self.lstm(x)

        x = x[:, -1, :]              # last timestep

        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))

        return x