import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable


class Segmentor(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        modules = list(resnet18.children())[:-4]
        resnet18 = nn.Sequential(*modules)
        for p in resnet18.parameters():
            p.requires_grad = False
        self.feature_extractor = resnet18
        self.conv1 = torch.nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            stride=1,
            bias=False,
            padding="same",
        )
        self.bn1 = torch.nn.BatchNorm2d(num_features=128)
        self.conv2 = torch.nn.Conv2d(
            in_channels=128,
            out_channels=256,
            kernel_size=7,
            stride=1,
            bias=False,
            padding="same",
        )
        self.bn2 = torch.nn.BatchNorm2d(num_features=256)
        self.conv3 = torch.nn.Conv2d(
            in_channels=256,
            out_channels=64,
            kernel_size=3,
            stride=1,
            bias=False,
            padding="same",
        )
        self.bn3 = torch.nn.BatchNorm2d(num_features=64)
        self.conv4 = torch.nn.Conv2d(
            in_channels=64,
            out_channels=1,
            kernel_size=1,
            stride=1,
            bias=False,
            padding="same",
        )
        self.bn4 = torch.nn.BatchNorm2d(num_features=1)
        self.act_func = torch.nn.ReLU6()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(input)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act_func(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act_func(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act_func(x)

        x = self.conv4(x)
        x = self.bn4(x)
        output = x
        return output
