import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels=64):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return out

class BaselineNet(nn.Module):
    def __init__(self):
        super(BaselineNet, self).__init__()
        self.head = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(8)])
        self.tail = nn.Conv2d(64, 3, kernel_size=3, padding=1)

    def forward(self, x):        
        out = self.head(x)
        out = self.blocks(out)
        out = self.tail(out)
        return x - out 