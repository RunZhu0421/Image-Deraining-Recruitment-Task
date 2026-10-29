# part 1
## 任务 3

### 环境配置
![环境配置](https://s41.ax1x.com/2026/10/29/pnydJHK.png)

### 实现基础模型
```
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
```

### 关于 CNN 感受野有限的思考
#### 可以使用空洞卷积，引入多尺度分支

### 运行代码（电脑带不动了所以就跑了三轮）
![运行](https://s41.ax1x.com/2026/10/29/pny0okT.png)

