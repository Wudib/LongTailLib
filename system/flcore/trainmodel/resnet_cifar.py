"""
ResNet for CIFAR (CReFF-style)
Adapted from CReFF-FL-main/Model/Resnet8.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def norm2d(group_norm_num_groups, planes):
    if group_norm_num_groups is not None and group_norm_num_groups > 0:
        return nn.GroupNorm(group_norm_num_groups, planes)
    else:
        return nn.BatchNorm2d(planes)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding."""
    return nn.Conv2d(
        in_channels=in_planes,
        out_channels=out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_planes,
        out_planes,
        stride=1,
        downsample=None,
        group_norm_num_groups=None,
    ):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, out_planes, stride)
        self.bn1 = norm2d(group_norm_num_groups, planes=out_planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(out_planes, out_planes)
        self.bn2 = norm2d(group_norm_num_groups, planes=out_planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNetBase(nn.Module):
    def _weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_block(
        self, block_fn, planes, block_num, stride=1, group_norm_num_groups=None
    ):
        downsample = None
        if stride != 1 or self.inplanes != planes * block_fn.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block_fn.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                norm2d(group_norm_num_groups, planes=planes * block_fn.expansion),
            )

        layers = []
        layers.append(
            block_fn(
                in_planes=self.inplanes,
                out_planes=planes,
                stride=stride,
                downsample=downsample,
                group_norm_num_groups=group_norm_num_groups,
            )
        )
        self.inplanes = planes * block_fn.expansion

        for _ in range(1, block_num):
            layers.append(
                block_fn(
                    in_planes=self.inplanes,
                    out_planes=planes,
                    group_norm_num_groups=group_norm_num_groups,
                )
            )
        return nn.Sequential(*layers)


class ResNet_cifar(ResNetBase):
    """
    ResNet for CIFAR datasets (CReFF-style)
    
    For resnet_size=8, scaling=4:
    - block_nums = 1
    - channels: 64 -> 128 -> 256
    - feature_dim = 256
    """
    def __init__(
        self,
        resnet_size=8,
        scaling=4,
        group_norm_num_groups=None,
        num_classes=10
    ):
        super(ResNet_cifar, self).__init__()

        # define Model.
        if resnet_size % 6 != 2:
            raise ValueError("resnet_size must be 6n + 2:", resnet_size)
        block_nums = (resnet_size - 2) // 6
        block_fn = BasicBlock

        self.num_classes = num_classes

        # define layers.
        assert int(16 * scaling) > 0
        self.inplanes = int(16 * scaling)
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=(16 * scaling),
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = norm2d(group_norm_num_groups, planes=int(16 * scaling))
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_block(
            block_fn=block_fn,
            planes=int(16 * scaling),
            block_num=block_nums,
            group_norm_num_groups=group_norm_num_groups,
        )
        self.layer2 = self._make_block(
            block_fn=block_fn,
            planes=int(32 * scaling),
            block_num=block_nums,
            stride=2,
            group_norm_num_groups=group_norm_num_groups,
        )
        self.layer3 = self._make_block(
            block_fn=block_fn,
            planes=int(64 * scaling),
            block_num=block_nums,
            stride=2,
            group_norm_num_groups=group_norm_num_groups,
        )

        self.avgpool = nn.AvgPool2d(kernel_size=8)
        self.classifier = nn.Linear(
            in_features=int(64 * scaling * block_fn.expansion),
            out_features=self.num_classes,
        )

        # weight initialization
        self._weight_initialization()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        feature = x
        y = self.classifier(x)

        return feature, y


def resnet8_cifar(num_classes=10, scaling=4):
    """
    ResNet8 for CIFAR (CReFF-style)
    
    Args:
        num_classes: number of classes
        scaling: channel scaling factor (default 4)
        
    Returns:
        ResNet_cifar model with feature_dim=256 (for scaling=4)
    """
    return ResNet_cifar(
        resnet_size=8,
        scaling=scaling,
        group_norm_num_groups=None,
        num_classes=num_classes
    )


class ResNet_cifar_512(ResNetBase):
    """
    ResNet for CIFAR with 512-dim features (CLIP2FL-style)
    
    Adds an MLP layer to map 256 -> 512 feature dimension.
    Reference: CLIP2FL-main/Model/Resnet8_256.py
    """
    def __init__(
        self,
        resnet_size=8,
        scaling=4,
        group_norm_num_groups=None,
        num_classes=10
    ):
        super(ResNet_cifar_512, self).__init__()

        if resnet_size % 6 != 2:
            raise ValueError("resnet_size must be 6n + 2:", resnet_size)
        block_nums = (resnet_size - 2) // 6
        block_fn = BasicBlock

        self.num_classes = num_classes

        assert int(16 * scaling) > 0
        self.inplanes = int(16 * scaling)
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=(16 * scaling),
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = norm2d(group_norm_num_groups, planes=int(16 * scaling))
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_block(
            block_fn=block_fn,
            planes=int(16 * scaling),
            block_num=block_nums,
            group_norm_num_groups=group_norm_num_groups,
        )
        self.layer2 = self._make_block(
            block_fn=block_fn,
            planes=int(32 * scaling),
            block_num=block_nums,
            stride=2,
            group_norm_num_groups=group_norm_num_groups,
        )
        self.layer3 = self._make_block(
            block_fn=block_fn,
            planes=int(64 * scaling),
            block_num=block_nums,
            stride=2,
            group_norm_num_groups=group_norm_num_groups,
        )

        self.avgpool = nn.AvgPool2d(kernel_size=8)
        
        # MLP layer: 256 -> 512 (CLIP2FL-style)
        self.add_mlp = nn.Linear(
            in_features=int(64 * scaling * block_fn.expansion),
            out_features=int(64 * scaling * 2 * block_fn.expansion),
        )
        self.classifier = nn.Linear(
            in_features=int(64 * scaling * 2 * block_fn.expansion),
            out_features=self.num_classes,
        )

        self._weight_initialization()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        # MLP layer
        x = self.add_mlp(x)

        feature = x
        y = self.classifier(x)

        return feature, y


def resnet8_cifar_512(num_classes=10, scaling=4):
    """
    ResNet8 for CIFAR with 512-dim features (CLIP2FL-style)
    
    Args:
        num_classes: number of classes
        scaling: channel scaling factor (default 4)
        
    Returns:
        ResNet_cifar_512 model with feature_dim=512 (for scaling=4)
    """
    return ResNet_cifar_512(
        resnet_size=8,
        scaling=scaling,
        group_norm_num_groups=None,
        num_classes=num_classes
    )


# ============================================================================
# FedLoGe-style ResNet18 for CIFAR
# Source: FedLoGe-master/model/model_res.py
# ============================================================================

import torch.nn.functional as F

class BasicBlock_FedLoGe(nn.Module):
    """
    BasicBlock for FedLoGe ResNet
    Source: FedLoGe-master/model/model_res.py line 11-33
    """
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock_FedLoGe, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet18_CIFAR(nn.Module):
    """
    ResNet18 for CIFAR-10/100 (512-dim features)
    Source: FedLoGe-master/model/model_res.py line 119-169
    
    Features:
    - 4 stages: 64 -> 128 -> 256 -> 512 channels
    - 512-dim feature output
    - Supports latent_output parameter for feature extraction
    - No maxpool (CIFAR-optimized)
    """
    def __init__(self, num_classes=10):
        super(ResNet18_CIFAR, self).__init__()
        self.in_planes = 64
        block = BasicBlock_FedLoGe
        num_blocks = [2, 2, 2, 2]

        # Stem: 3x3 conv, no maxpool (CIFAR-optimized)
        self.conv1 = conv3x3(3, 64)
        self.bn1 = nn.BatchNorm2d(64)
        
        # 4 stages
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, latent_output=False):
        """
        Forward pass with optional feature extraction
        
        Args:
            x: input tensor
            latent_output: if True, return 512-dim features; if False, return logits
        """
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        
        if latent_output:
            return out  # 512-dim features
        else:
            return self.linear(out)  # logits


def resnet18_cifar(num_classes=10):
    """
    ResNet18 for CIFAR-10/100 (512-dim features)
    
    Args:
        num_classes: number of classes
        
    Returns:
        ResNet18_CIFAR model with feature_dim=512
    """
    return ResNet18_CIFAR(num_classes=num_classes)
