"""
  Filename    [ improve_models.py ]
  PackageName [ YOLOv1 ]
  Synposis    [ Deprecated: To be moved to model.py ]
"""

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torchvision.models
           
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class ResidualBlocks(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, downsample=None, groups=1, base_width=64, norm_layer=None):
        super(ResidualBlocks, self).__init__()

        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride)
        self.bn1   = nn.BatchNorm2d(out_channel)
        self.relu  = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3)
        self.bn2   = nn.BatchNorm2d(out_channel)
        self.relu  = nn.ReLU(inplace=True)
        self.drop  = nn.Dropout2d(0.5, inplace=False)

        self.stride = stride
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.drop(out)
        
        out += identity
        out = self.relu(out)

        return out

class Yolov1_ResNet(nn.Module):
    def __init__(self, blocks):
        super(Yolov1_ResNet, self).__init__()

        # Implement ResNet Structure
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2)
        self.bn1   = nn.BatchNorm2d(num_features=64)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.ResidualBlocks = blocks

        # self.rb1 = ResidualBlocks(64, 64, stride=2)
        # self.rb2 = ResidualBlocks(64, 64)
        # self.rb3 = ResidualBlocks(64, 128, stride=2)
        # self.rb4 = ResidualBlocks(128, 128)
        # self.rb5 = ResidualBlocks(128, 256, stride=2)
        # self.rb6 = ResidualBlocks(256, 256)
        # self.rb7 = ResidualBlocks(256, 512, stride=2)
        # self.rb8 = ResidualBlocks(512, 512)

        # Implement Yolo detection
        self.yolo = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 1274)
        )

    def forward(self, x):
        """
          input_size:    n * 3 * 448 * 448
          VGG16_bn:      n * 512 * 7 * 7
          Flatten Layer: n * 25088
          Yolo Layer:    n * 1274
          Sigmoid Layer: n * 1274
          Reshape Layer: n * 26 * 7 * 7
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool(x)
        
        for block in self.ResidualBlocks:
            x = block(x)
            # print(x.shape, x.dtype)
        
        x = x.view(x.size(0), -1)
        # print(x.shape, x.dtype)
        
        x = self.yolo(x)
        # print(x.shape, x.dtype)
        
        x = torch.sigmoid(x) 
        # print(x.shape, x.dtype)
        
        x = x.view(-1, 7, 7, 26)
        # print(x.shape, x.dtype)
        
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()