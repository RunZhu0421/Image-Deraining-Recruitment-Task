import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels):
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

class DerainNet(nn.Module):
    def __init__(self):
        super(DerainNet, self).__init__()
        # 基础特征提取
        self.head = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        
        # 三个尺度的分支 (每个分支8个残差块)
        self.branch_1x = nn.Sequential(*[ResidualBlock(64) for _ in range(8)])
        self.branch_05x = nn.Sequential(*[ResidualBlock(64) for _ in range(8)])
        self.branch_025x = nn.Sequential(*[ResidualBlock(64) for _ in range(8)])
        
        # 融合层：输入是 3 个分支拼接的通道数 (64*3 = 192)
        self.fusion = nn.Conv2d(192, 3, kernel_size=3, padding=1)

    def forward(self, x):
        # 1. 提取初始特征
        feat = self.head(x)  # [B, 64, H, W]
        
        # 2. 下采样构建多尺度
        feat_05x = F.max_pool2d(feat, kernel_size=2, stride=2)       # 1/2 大小
        feat_025x = F.max_pool2d(feat_05x, kernel_size=2, stride=2)  # 1/4 大小
        
        # 3. 分别经过残差块，并加上原始特征（残差思想）
        out_1x = self.branch_1x(feat) + feat
        out_05x = self.branch_05x(feat_05x) + feat_05x
        out_025x = self.branch_025x(feat_025x) + feat_025x
        
        # 4. 将 1/2 和 1/4 的特征上采样回原图尺寸
        up_05x = F.interpolate(out_05x, size=out_1x.size()[2:], mode='bilinear', align_corners=False)
        up_025x = F.interpolate(out_025x, size=out_1x.size()[2:], mode='bilinear', align_corners=False)
        
        # 5. 三路 Concat 拼接（在通道维度拼接）
        concat_feat = torch.cat([out_1x, up_05x, up_025x], dim=1) # [B, 192, H, W]
        
        # 6. 卷积融合输出雨层
        rain_layer = self.fusion(concat_feat)
        
        # 7. 残差学习返回无雨图
        return x - rain_layer