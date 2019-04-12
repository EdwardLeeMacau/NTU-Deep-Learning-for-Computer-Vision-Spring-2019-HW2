import torch.nn as nn
import math
import torch
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import utils
from torch.autograd import Variable

__all__ = ['vgg16_bn']
model_urls = {
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
}

class VGG(nn.Module):
    """
      input_size  = 448 * 448
      output_size = 7 * 7 * (5 * 2 + 16) = 1274
    """
    def __init__(self, features, output_size=1274, image_size=448):
        super(VGG, self).__init__()
        self.features = features
        self.image_size = image_size

        self.yolo = nn.Sequential(
            #TODO
            nn.Linear(25088, 4096), 
            nn.Linear(4096, 1274)
        )
        self._initialize_weights()

    def forward(self, x):
        """
          input_size:    n * 3 * 448 * 448
          VGG16_bn:      n * 512 * 7 * 7
          Flatten Layer: n * 25088
          Yolo Layer:    n * 1274
          Sigmoid Layer: n * 1274
          Reshape Layer: n * 26 * 7 * 7
        """
        # print(x.shape, x.dtype)
        
        x = self.features(x)
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

class YoloLoss(nn.Module):
    def __init__(self, grid_num, bbox_num, lambda_coord, lambda_noobj):
        super(YoloLoss, self).__init__()
        self.grid_num = grid_num
        self.bbox_num = bbox_num
        self.lambda_coord = lambda_noobj
        self.lambda_noobj = lambda_noobj

    def IoU(self, tensor1, tensor2):
        """
        Args:
          tensor1: [num_bbox, 10]
          tensor2: [num_bbox, 10]
        """

        # print("Tensor1.shape: {}".format(tensor1.shape))

        num_bbox = tensor1.shape[0]

        intersectionArea = torch.zeros(num_bbox, 2)
        left_top     = torch.zeros(num_bbox, 4)
        right_bottom = torch.zeros(num_bbox, 4)

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

        inter_wh = right_bottom - left_top
        inter_wh[inter_wh < 0] = 0
        intersectionArea[:, 0] = inter_wh[:, 0] * inter_wh[:, 1]
        intersectionArea[:, 1] = inter_wh[:, 2] * inter_wh[:, 3]

        area_1_1 = (tensor1[:, 2] - tensor1[:, 0]) * (tensor1[:, 3] - tensor1[:, 1])
        area_1_2 = (tensor1[:, 7] - tensor1[:, 5]) * (tensor1[:, 8] - tensor1[:, 6])
        # print("area_1_1.shape: {}".format(area_1_1.shape))
        # print("area_1_2.shape: {}".format(area_1_2.shape))
        area_1  = torch.cat((area_1_1.unsqueeze(1), area_1_2.unsqueeze(1)), dim=1)
        # print("area_1.shape: {}".format(area_1.shape))
        area_2_1 = (tensor2[:, 2] - tensor2[:, 0]) * (tensor2[:, 3] - tensor2[:, 1])
        area_2_2 = (tensor2[:, 7] - tensor2[:, 5]) * (tensor2[:, 8] - tensor2[:, 6])
        area_2  = torch.cat((area_2_1.unsqueeze(1), area_2_2.unsqueeze(1)), dim=1)
        # print("area_2.shape: {}".format(area_2.shape))

        iou = intersectionArea / (area_1 + area_2 - intersectionArea)

        return iou
        
    def nonMaximumSuppression(self, boxes):
        return

    def forward(self, output: torch.tensor, target: torch.tensor):
        """
        Args:
          output: [batchsize, 7, 7, 26]
          target: [batchsize, 7, 7, 26]

        Output:
          loss
        """
        loss = 0
        batch_size = output.shape[0]

        coord_mask = (target[:, :, :, 4] > 0).unsqueeze(-1).expand_as(target)
        noobj_mask = (target[:, :, :, 4] == 0).unsqueeze(-1).expand_as(target)

        coord_predict = output[coord_mask].view(-1, 26)
        coord_target  = target[coord_mask].view(-1, 26)
        noobj_predict = output[noobj_mask].view(-1, 26)
        noobj_target  = output[noobj_mask].view(-1, 26)

        """
        boxes_predict = coord_predict[:, :10]
        class_predict = coord_predict[:, 10:]

        boxes_target  = target[:, :10]
        class_target  = target[:, 10:]
        """

        # Compute the loss of not-containing object
        noobj_predict_confidence = torch.cat((noobj_predict[:, 4], noobj_predict[:, 9]), dim=1)
        noobj_target_confidence  = torch.cat((noobj_target[:, 4], noobj_target[:, 9]), dim=1)

        loss += self.lambda_noobj * F.mse_loss(noobj_predict_confidence, noobj_target_confidence, size_average=False)

        # Compute the loss of containing object
        boxes_predict = coord_predict[:, :10]       # Match "delta_xy" in dataset.py
        boxes_target  = coord_target[:, :10]        # Match "delta_xy" in dataset.py
        
        N = boxes_predict.shape[0]                  # N: the number of bbox predicted
        # print("Number of boxes: {}".format(boxes_predict.shape))
        # print("Number of boxes: {}".format(boxes_target.shape))
        boxes_predict_xy = torch.zeros(N, 10)
        boxes_target_xy  = torch.zeros(N, 10)
        # print("Boxes_predict_xy.shape: {}".format(boxes_predict_xy.shape))
        # print("Boxes_predict.shape: {}".format(boxes_predict.shape))

        boxes_predict_xy[:,  :2] = boxes_predict[:,  :2] / 7 - 0.5 * boxes_predict[:, 2:4]
        boxes_predict_xy[:, 2:4] = boxes_predict[:,  :2] / 7 + 0.5 * boxes_predict[:, 2:4]
        boxes_predict_xy[:, 5:7] = boxes_predict[:, 5:7] / 7 - 0.5 * boxes_predict[:, 7:9]
        boxes_predict_xy[:, 7:9] = boxes_predict[:, 5:7] / 7 + 0.5 * boxes_predict[:, 7:9]
        boxes_predict_xy[:, 4], boxes_predict_xy[:, 9] = boxes_predict[:, 4], boxes_predict[:, 9]

        boxes_target_xy[:,  :2] = boxes_target[:,  :2] / 7 - 0.5 * boxes_target[:, 2:4]
        boxes_target_xy[:, 2:4] = boxes_target[:,  :2] / 7 + 0.5 * boxes_target[:, 2:4]
        boxes_target_xy[:, 5:7] = boxes_target[:, 5:7] / 7 - 0.5 * boxes_target[:, 7:9]
        boxes_target_xy[:, 7:9] = boxes_target[:, 5:7] / 7 + 0.5 * boxes_target[:, 7:9]
        boxes_target_xy[:, 4], boxes_target_xy[:, 9] = boxes_target[:, 4], boxes_target[:, 9]
        
        iou = self.IoU(boxes_predict_xy, boxes_target_xy)
        # print("IoU.shape: {}".format(iou.shape))
        iou_max, max_index = iou.max(dim=1)
        print("IoU_Max.shape: {}".format(iou_max.shape))
        print("IoU_Max: {}".format(iou_max))
        print("max_index.shape: {}".format(max_index.shape))
        print("max_index: {}".format(max_index))

        coord_response_mask = torch.zeros_like(boxes_target)
        coord_not_response_mask = torch.zeros_like(boxes_target)
        coord_response_mask[max_index] = 1
        coord_not_response_mask[1 - max_index] = 1
        
        boxes_target_iou = torch.zeros_like(boxes_target)
        boxes_target_iou[max_index] = iou_max
        boxes_target_iou = Variable(boxes_target_iou)
        
        # Compute the loss of class loss
        loss += F.mse_loss(output[:, :, 10:], target[:, :, 10:], size_average=False)

        return loss

# Using the configuration to make the layers
def make_layers(cfg, batch_norm=False):
    """
    Args:
        cfg: the sequence configuration with ints and chars.
        batch_norm: provide batch normalization layer

    Return:
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
}


def Yolov1_vgg16bn(pretrained=False, **kwargs):
    """
    VGG 16-layer model (configuration "D") with batch normalization
    
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
            
    Return:
        yolo: the prediction model YOLO.
    """

    # print(make_layers(cfg['D'], batch_norm=True))

    yolo = VGG(make_layers(cfg['D'], batch_norm=True), **kwargs)

    vgg_state_dict = model_zoo.load_url(model_urls['vgg16_bn'])
    yolo_state_dict = yolo.state_dict()
    for k in vgg_state_dict.keys():
        if k in yolo_state_dict.keys() and k.startswith('features'):
            yolo_state_dict[k] = vgg_state_dict[k]
    
    yolo.load_state_dict(yolo_state_dict)
    
    return yolo

def model_structure_unittest():
    model = Yolov1_vgg16bn(pretrained=True)
    # print(model)

    img = torch.rand(1, 3, 448, 448)
    output = model(img)
    # print(output.size())
    criterion = YoloLoss(7, 2, 5, 0.5)
    target = torch.rand(1, 7, 7, 26)
    loss = criterion(output, target)

    print(loss)

if __name__ == '__main__':
    model_structure_unittest()
