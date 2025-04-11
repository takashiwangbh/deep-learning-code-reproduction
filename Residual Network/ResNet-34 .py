import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义基本的残差块：BasicBlock
class BasicBlock(nn.Module):
    expansion = 1  # 通道数扩张倍数，ResNet-34中为1，ResNet-50中为4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        """
            in_channels: 输入通道数
            out_channels: 输出通道数
            stride: 步长，控制下采样
            downsample: 是否需要下采样，用于匹配维度
        """
        super(BasicBlock, self).__init__()

        # 第一个卷积层
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,stride=stride, padding=1, bias=False) # 创建一个 3x3 的卷积层，输入通道是 in_channels，输出是 out_channels
        self.bn1 = nn.BatchNorm2d(out_channels) # 批归一化层（Batch Normalization），用于加速训练

        # 第二个卷积层（注意stride=1）
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                            stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 下采样模块（若需要）
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True) # ReLU 激活函数，输出变非线性，让网络能学更复杂的东西

    def forward(self, x):
        identity = x  # 保存原始输入用于残差连接

        # 主路径：卷积 -> BN -> ReLU -> 卷积 -> BN
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 如果输入和输出维度不一样，就用一个卷积调整它（就是“跳连”那条线）
        if self.downsample is not None:
            identity = self.downsample(x)

        # 残差连接
        out += identity
        out = self.relu(out)

        return out

# 定义 ResNet-34 网络
class ResNet34(nn.Module):
    def __init__(self, num_classes=1000):
        """
        Args:
            num_classes: 分类类别数
        """
        super(ResNet34, self).__init__()
        self.in_channels = 64

        # 第一层：初始卷积层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 输入图像是彩色图（3通道），输出是 64 个特征图 7x7 是卷积核大小，stride=2 表示缩小尺寸（下采样）
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # 池化层
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 池化层：用来进一步缩小图片尺寸（减小计算量） 输出: 64 x 56 x 56

        # 四个stage（每个stage内部包含多个BasicBlock）
        self.layer1 = self._make_layer(64, 3)     # 输出: 64 x 56 x 56
        self.layer2 = self._make_layer(128, 4, stride=2)  # 输出: 128 x 28 x 28
        self.layer3 = self._make_layer(256, 6, stride=2)  # 输出: 256 x 14 x 14
        self.layer4 = self._make_layer(512, 3, stride=2)  # 输出: 512 x 7 x 7

        # 全局平均池化 + 全连接层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 输出: 512 x 1 x 1
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, out_channels, blocks, stride=1):
        """
        构造一个残差块序列（一个stage）
        Args:
            out_channels: 该层的输出通道数
            blocks: block 数量
            stride: 第一块的步长，控制下采样
        """
        downsample = None
        layers = []

        # 若输入通道 != 输出通道 或 stride != 1，则需要下采样
        if stride != 1 or self.in_channels != out_channels * BasicBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * BasicBlock.expansion,
                        kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

        # 第一块可能需要下采样
        layers.append(BasicBlock(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * BasicBlock.expansion

        # 后续 blocks 步长都为 1
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        # 初始处理：卷积 -> BN -> ReLU -> 最大池化
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # 4个残差层
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # 全局平均池化 + 全连接分类
        x = self.avgpool(x)         # [B, 512, 1, 1]
        x = torch.flatten(x, 1)     # [B, 512]
        x = self.fc(x)              # [B, num_classes]

        return x
