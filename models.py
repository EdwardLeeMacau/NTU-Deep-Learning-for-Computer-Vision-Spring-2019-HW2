import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torchvision.models
import utils

__all__ = ['vgg16_bn']

model_urls = {
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

model_urls = {

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
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool(x)
        
        for block in self.ResidualBlocks:
            x = block(x)
        
        x = x.view(x.size(0), -1)
        x = self.yolo(x)
        x = torch.sigmoid(x)
        x = x.view(-1, 7, 7, 26)
 
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

class VGG(nn.Module):
    def __init__(self, features, output_size=1274, image_size=448):
        super(VGG, self).__init__()
        self.features = features
        self.image_size = image_size

        self.yolo = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(4096, 1274)
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x).view(x.size(0), -1)        
        x = self.yolo(x)
        x = torch.sigmoid(x) 
        x = x.view(-1, 7, 7, 26)
        
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

class VGG_Improve(nn.Module):
    def __init__(self, features, output_size=5096, image_size=448):
        super(VGG_Improve, self).__init__()
        self.features = features
        self.image_size = image_size

        self.yolo = nn.Sequential(
            nn.Conv2d(512, 26, kernel_size=1),
            # nn.Linear(25088, 8192),
            # nn.BatchNorm1d(num_features=4096),
            # nn.LeakyReLU(negative_slope=0.02),
            # nn.ReLU(inplace=True),
            # nn.Dropout(0.5),

            # nn.Linear(8192, 8192),
            # nn.BatchNorm1d(num_features=8192),
            # nn.ReLU(negative_slope=0.02),
            # nn.Dropout(0.5),

            # nn.Linear(8192, 5096)
        )

        self.bn = nn.BatchNorm2d(num_features=26)

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.yolo(x)
        x = self.bn(x)
        x = torch.sigmoid(x) 
        x = x.permute(0, 2, 3, 1)
        
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

class YoloLoss(nn.Module):
    def __init__(self, grid_num, bbox_num, lambda_coord, lambda_noobj, device):
        super(YoloLoss, self).__init__()
        self.grid_num = grid_num
        self.bbox_num = bbox_num
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.device = device

    def IoU(self, tensor1, tensor2):
        """
        Calculate the Intersection of Union

        Parameters
        ----------
        tensor1, tensor2: torch.tensor
            [num_bbox, 10] [num_bbox, 10]
        """
        tensor1 = tensor1.type(torch.float)
        tensor2 = tensor2.type(torch.float)

        num_bbox = tensor1.shape[0]

        intersectionArea = torch.zeros(num_bbox, 2).to(self.device)
        left_top     = torch.zeros(num_bbox, 4).to(self.device)
        right_bottom = torch.zeros(num_bbox, 4).to(self.device)

        left_top[:, :2] = torch.max(
            tensor1[:, :2],
            tensor2[:, :2]
        )
        left_top[:, 2:] = torch.max(
            tensor1[:, 5:7],
            tensor2[:, 5:7]
        )

        right_bottom[:, :2] = torch.min(
            tensor1[:, 2:4],
            tensor2[:, 2:4]
        )
        right_bottom[:, 2:] = torch.min(
            tensor1[:, 7:9],
            tensor2[:, 7:9]
        )

        inter_wh = (right_bottom - left_top).clamp(min=0)
        intersectionArea[:, 0] = inter_wh[:, 0] * inter_wh[:, 1]
        intersectionArea[:, 1] = inter_wh[:, 2] * inter_wh[:, 3]

        area_1_1 = (tensor1[:, 2] - tensor1[:, 0]) * (tensor1[:, 3] - tensor1[:, 1])
        area_1_2 = (tensor1[:, 7] - tensor1[:, 5]) * (tensor1[:, 8] - tensor1[:, 6])
        area_1  = torch.cat((area_1_1.unsqueeze(1), area_1_2.unsqueeze(1)), dim=1).to(self.device)
        
        area_2_1 = (tensor2[:, 2] - tensor2[:, 0]) * (tensor2[:, 3] - tensor2[:, 1])
        area_2_2 = (tensor2[:, 7] - tensor2[:, 5]) * (tensor2[:, 8] - tensor2[:, 6])
        area_2  = torch.cat((area_2_1.unsqueeze(1), area_2_2.unsqueeze(1)), dim=1).to(self.device)
        
        iou = intersectionArea / (area_1 + area_2 - intersectionArea)

        return iou

    def xyhw_xyxy(self, tensor: torch.tensor):
        tensor_xy = torch.zeros_like(tensor)
        
        tensor_xy[:,  :2] = tensor[:,  :2] / self.grid_num - 0.5 * tensor[:, 2:4]
        tensor_xy[:, 2:4] = tensor[:,  :2] / self.grid_num + 0.5 * tensor[:, 2:4]
        tensor_xy[:, 5:7] = tensor[:, 5:7] / self.grid_num - 0.5 * tensor[:, 7:9]
        tensor_xy[:, 7:9] = tensor[:, 5:7] / self.grid_num + 0.5 * tensor[:, 7:9]
        tensor_xy[:, 4], tensor_xy[:, 9] = tensor[:, 4], tensor[:, 9]

        return tensor_xy

    def forward(self, output: torch.tensor, target: torch.tensor):
        """
        Default using cuda speedup.

        The Loss of YOLO can be devided as 5 parts:
        1. Class Loss
        2. No Object Loss
        3. Object Loss
        4. Location Loss
        5. Not Response Loss

        Parameters
        ----------
        output, target: torch.tensor
            [batchsize, 7, 7, 26]
          
        Return
        ------
        loss: 
            (...)
        """
        loss = 0
        batch_size = output.shape[0]

        # Assumption 1: the gt contain 1 object only
        coord_mask = (target[:, :, :, 4] > 0).unsqueeze(-1).expand_as(target)
        noobj_mask = (target[:, :, :, 4] == 0).unsqueeze(-1).expand_as(target)

        coord_predict = output[coord_mask].view(-1, 26).type(torch.float)
        coord_target  = target[coord_mask].view(-1, 26).type(torch.float)
        noobj_predict = output[noobj_mask].view(-1, 26).type(torch.float)
        noobj_target  = target[noobj_mask].view(-1, 26).type(torch.float)

        """ Loss 1: Class_loss """
        class_loss = F.mse_loss(coord_predict[:, 10:], coord_target[:, 10:], size_average=False)

        """ Loss 2: No_object_Loss """
        no_object_loss = (F.mse_loss(noobj_predict[:, 4], noobj_target[:, 4], size_average=False)
                         + F.mse_loss(noobj_predict[:, 9], noobj_target[:, 9], size_average=False))

        # 2. Compute the loss of containing object
        boxes_predict = coord_predict[:, :10]       # Match "delta_xy" in dataset.py
        boxes_target  = coord_target[:, :10]        # Match "delta_xy" in dataset.py
        
        boxes_predict_xy = self.xyhw_xyxy(boxes_predict)
        boxes_target_xy = self.xyhw_xyxy(boxes_target)
        iou = self.IoU(boxes_predict_xy, boxes_target_xy)
        # print("IoU: {}".format(iou))
        # print("IoU.shape: {}".format(iou.shape))
        # print("Iou: {}".format(iou))
        iou_max, max_index = iou.max(dim=1)
        max_index = max_index.type(torch.uint8)
        min_index = max_index.le(0)
        # print("IoU_Max.shape: {}".format(iou_max.shape))
        # print("IoU_Max: {}".format(iou_max))
        # print("max_index.shape: {}".format(max_index.shape))
        # print("max_index: {}".format(max_index))
        # print("max_index: {}".format(max_index.dtype))

        # Response Mask: the mask that notes the box need to calculate position loss.
        response_mask = torch.zeros((iou_max.shape[0], 2), dtype=torch.uint8)
        response_mask[max_index, 1] = 1
        response_mask[min_index, 0] = 1
        # coord_response_mask = coord_response_mask.view(-1, 2)
        response_mask = response_mask.view(-1)
        not_response_mask = response_mask.le(0)
        # print("coord_response_mask.shape: {}".format(coord_response_mask.shape))
        # print("coord_response_mask: {}".format(coord_response_mask))
        # print("coord_not_response_mask.shape: {}".format(coord_not_response_mask.shape))
        # print("coord_not_response_mask: {}".format(coord_not_response_mask))

        boxes_predict = boxes_predict.contiguous().view(-1, 5)
        boxes_target_iou = boxes_target.type(torch.float).contiguous().view(-1, 5)
        boxes_target_iou[response_mask, 4] = iou_max
        # boxes_target_iou[response_mask, 4] = 1
        boxes_target_iou[not_response_mask, 4] = 0
        # print("boxes_target_iou.shape: {}".format(boxes_target_iou.shape))
        # print("boxes_target_iou: {}".format(boxes_target_iou))

        boxes_predict_response = boxes_predict[response_mask]
        boxes_target_response  = boxes_target_iou[response_mask]
        # print("Boxes_predict_response.shape: {}".format(boxes_predict_response.shape))
        # print("Boxes_target_response.shape: {}".format(boxes_predict_response.shape))
        # print("Boxes_predict_response: {}".format(boxes_predict_response))
        # print("Boxes_target_response: {}".format(boxes_target_response))
        # boxes_target_response = boxes_target[coord_response_mask].view(-1, 5)

        """ Class 3: Contain_loss """
        response_loss = F.mse_loss(boxes_predict_response[:, 4], boxes_target_response[:, 4], size_average=False)
        
        """ Class 4: Location_loss """
        location_loss = (F.mse_loss(boxes_predict_response[:, :2], boxes_target_response[:, :2], size_average=False) + 
                         F.mse_loss(torch.sqrt(boxes_predict_response[:, 2:4]), torch.sqrt(boxes_target_response[:, 2:4]), size_average=False))
        
        # 2.2 not response loss, set the gt of the confidence as 0
        boxes_predict_not_response = boxes_predict[not_response_mask]
        boxes_target_not_response  = boxes_target_iou[not_response_mask]

        # print("noobj_confidence_loss: {}".format(self.lambda_noobj * F.mse_loss(boxes_predict_not_response[:, 4], boxes_target_not_response[:, 4], size_average=False)))
        """ Class 5: Not_response_loss """
        not_response_loss = F.mse_loss(boxes_predict_not_response[:, 4], boxes_target_not_response[:, 4], size_average=False)
        
        # Output the normalized loss
        loss = self.lambda_coord * location_loss + class_loss + response_loss + self.lambda_noobj * (not_response_loss + no_object_loss)
        loss /= batch_size
        return loss

# Using the configuration to make the layers
def make_layers(cfg, batch_norm=False):
    """
    Parameters
    ----------
    cfg: the sequence configuration with ints and chars.
    
    batch_norm: provide batch normalization layer

    Return
    ------
    model : nn.Module
        nn.Sequential(*layers): the model sequence
    """
    layers = []
    in_channels = 3
    s = 1
    first_flag=True

    for v in cfg:
        s = 1
        
        # Only the first_flag should set stride = 2
        if (v == 64 and first_flag):
            s = 2
            first_flag = False
        
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, stride=s, padding=1)
            
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            
            in_channels = v

    return nn.Sequential(*layers)

def conv_bn_relu(in_channels,out_channels,kernel_size=3,stride=2,padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,padding=padding,stride=stride),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(True)
    )


# Configuration of VGG
# number = the number of output channels of the conv layer
#    "M" = MaxPooling Layer
cfg = {
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'D2': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]
}

def Yolov1_vgg16bn(pretrained=False, **kwargs):
    """
    VGG 16-layer model (configuration "D") with batch normalization
    
    Parameters
    ----------
    pretrained : bool
        If True, returns a model pre-trained on ImageNet
        
    Return
    ------
    yolo: 
        The prediction model YOLO.
    """
    yolo = VGG(make_layers(cfg['D'], batch_norm=True), **kwargs)

    if pretrained:
        vgg_state_dict = model_zoo.load_url(model_urls['vgg16_bn'])
        yolo_state_dict = yolo.state_dict()
        for k in vgg_state_dict.keys():
            if k in yolo_state_dict.keys() and k.startswith('features'):
                yolo_state_dict[k] = vgg_state_dict[k]
        
        yolo.load_state_dict(yolo_state_dict)
    
    return yolo

def Yolov1_vgg16bn_Improve(pretrained=False, **kwargs):
    """
    VGG 16-layer model (configuration "D") with batch normalization
    
    Parameters
    ----------
    pretrained : bool: 
        If True, returns a model pre-trained on ImageNet
            
    Return
    ------
    yolo: nn.Module
        the prediction model YOLO.
    """

    # print(make_layers(cfg['D'], batch_norm=True))

    yolo = VGG_Improve(make_layers(cfg['D2'], batch_norm=True), **kwargs)

    vgg_state_dict = model_zoo.load_url(model_urls['vgg16_bn'])
    yolo_state_dict = yolo.state_dict()
    for k in vgg_state_dict.keys():
        if k in yolo_state_dict.keys() and k.startswith('features'):
            yolo_state_dict[k] = vgg_state_dict[k]
    
    yolo.load_state_dict(yolo_state_dict)
    
    return yolo
