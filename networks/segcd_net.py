import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple, Type, Union, List
from torchvision import models
from torchvision.models.resnet import BasicBlock, Bottleneck
from . import utils_resnet
from einops import rearrange


class ResnetFeatures(models.ResNet):
    
    def __init__(self, 
                 block: Type[Union[BasicBlock, Bottleneck]], 
                 layers: List[int], 
                 num_classes: int = 1000) -> None:
        super(ResnetFeatures, self).__init__(block, layers, num_classes)


    def set_parameter_requires_grad(self, feature_extracting : bool = True) -> None:
        if feature_extracting:
            for param in self.parameters():
                param.requires_grad = not feature_extracting


    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        # layer4 = self.avgpool(layer4)
        # layer4 = layer4.reshape(layer4.size(0), -1)
        # x = self.fc(x)
        return layer4

class ConvAndUpsample(nn.Module):
    def __init__(self, channel) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channel, channel // 2, kernel_size=3, stride=1),
            nn.BatchNorm2d(channel // 2),
            nn.LeakyReLU(),
            nn.Conv2d(channel // 2, channel // 2, kernel_size=1, stride=1),
            nn.BatchNorm2d(channel // 2),
            nn.LeakyReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.ConvTranspose2d(channel // 2, channel // 2, kernel_size=5),
            nn.LeakyReLU()
        )
    
    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)

class ASPPModule(nn.Module):
    def __init__(self, in_channels, out_channels, rates):
        super(ASPPModule, self).__init__()
        self.conv1x1_1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv3x3_2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rates[0], dilation=rates[0])
        self.conv3x3_3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rates[1], dilation=rates[1])
        self.conv3x3_4 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rates[2], dilation=rates[2])
        self.conv1x1_5 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.batch_norm = nn.BatchNorm2d(out_channels * 5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.conv1x1_1(x)
        x2 = self.conv3x3_2(x)
        x3 = self.conv3x3_3(x)
        x4 = self.conv3x3_4(x)
        x5 = self.conv1x1_5(x)
        out = torch.cat([x1, x2, x3, x4, x5], dim=1)
        out = self.batch_norm(out)
        out = self.relu(out)
        return out

class SegCDNet(nn.Module):
    def __init__(self, num_classes=7) -> None:
        super().__init__()
        self.backbone = utils_resnet.resnet34(ResnetFeatures, True)
        
        backbone_channel = 512
        
        self.transformer = nn.Transformer(d_model=backbone_channel, nhead=8)
        
        self.aspp = ASPPModule(backbone_channel, 256, [6, 12, 18])
        
        conv_channel = 2048
        self.upsample = nn.Sequential(
            nn.Conv2d(256 * 5, 1024, kernel_size=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            *[ConvAndUpsample(conv_channel // (2 ** (i + 1))) for i in range(5)]
        )
        self.classifer = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.SELU(),
            nn.Conv2d(32, num_classes, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_classes),
            nn.SELU(),
            nn.Conv2d(num_classes, num_classes, kernel_size=1),
            nn.BatchNorm2d(num_classes),
            nn.LeakyReLU()
        )
        

    
    def forward(self, t: Tensor) -> Tensor:  
        f: Tensor = self.backbone(t)
        img_size = f.shape[-1]
        t_i = rearrange(f, 'b c h w -> (h w) b c')
        t_o = self.transformer(t_i, t_i) + t_i
        t_o = rearrange(t_o, '(h w) b c -> b c h w', h=img_size)
        a_o = self.aspp(t_o)
        u_o = self.upsample(a_o)
        o = self.classifer(u_o)
        return o

# if __name__ == '__main__':
#     from torchinfo import summary
#     summary(SegCDNet(), (4, 3, 256, 256))