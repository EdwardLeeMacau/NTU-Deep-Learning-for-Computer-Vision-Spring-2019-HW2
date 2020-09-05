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
    def __init__(self, grid_num, bbox_num, lambda_coord, lambda_noobj, device):
        super(YoloLoss, self).__init__()
        self.grid_num = grid_num
        self.bbox_num = bbox_num
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.device = device

    def IoU(self, tensor1, tensor2):
        """
        Args:
          tensor1: [num_bbox, 10]
          tensor2: [num_bbox, 10]
        """
        tensor1 = tensor1.type(torch.float)
        tensor2 = tensor2.type(torch.float)
        # print("Tensor1.shape: {}".format(tensor1.shape))

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

        inter_wh = (right_bottom - left_top).to(self.device)
        inter_wh[inter_wh < 0] = 0
        intersectionArea[:, 0] = inter_wh[:, 0] * inter_wh[:, 1]
        intersectionArea[:, 1] = inter_wh[:, 2] * inter_wh[:, 3]

        area_1_1 = (tensor1[:, 2] - tensor1[:, 0]) * (tensor1[:, 3] - tensor1[:, 1])
        area_1_2 = (tensor1[:, 7] - tensor1[:, 5]) * (tensor1[:, 8] - tensor1[:, 6])
        # print("area_1_1.shape: {}".format(area_1_1.shape))
        # print("area_1_2.shape: {}".format(area_1_2.shape))
        area_1  = torch.cat((area_1_1.unsqueeze(1), area_1_2.unsqueeze(1)), dim=1).to(self.device)
        # print("area_1.shape: {}".format(area_1.shape))
        area_2_1 = (tensor2[:, 2] - tensor2[:, 0]) * (tensor2[:, 3] - tensor2[:, 1])
        area_2_2 = (tensor2[:, 7] - tensor2[:, 5]) * (tensor2[:, 8] - tensor2[:, 6])
        area_2  = torch.cat((area_2_1.unsqueeze(1), area_2_2.unsqueeze(1)), dim=1).to(self.device)
        # print("area_2.shape: {}".format(area_2.shape))

        iou = intersectionArea / (area_1 + area_2 - intersectionArea)

        return iou

    def forward(self, output: torch.tensor, target: torch.tensor):
        """
        Default using cuda speedup.

        Assumptions:
          1. the gt contain 1 object only
          2. 

        Args:
          output: [batchsize, 7, 7, 26]
          target: [batchsize, 7, 7, 26]

        Output:
          loss: scalar
        """
        loss = 0
        batch_size = output.shape[0]

        # Assumption 1: the gt contain 1 object only
        coord_mask = (target[:, :, :, 4] > 0).unsqueeze(-1).expand_as(target)
        noobj_mask = (target[:, :, :, 4] == 0).unsqueeze(-1).expand_as(target)

        coord_predict = output[coord_mask].view(-1, 26)
        coord_target  = target[coord_mask].view(-1, 26)
        noobj_predict = output[noobj_mask].view(-1, 26)
        noobj_target  = output[noobj_mask].view(-1, 26)

        # 1. Compute the loss of not-containing object
        noobj_predict_confidence = torch.cat((noobj_predict[:, 4].unsqueeze(1), noobj_predict[:, 9].unsqueeze(1)), dim=1)
        noobj_target_confidence  = torch.cat((noobj_target[:, 4].unsqueeze(1), noobj_target[:, 9].unsqueeze(1)), dim=1)

        # print("noobj_confidence_loss: {}".format(self.lambda_noobj * F.mse_loss(noobj_predict_confidence, noobj_target_confidence, size_average=False)))
        loss += self.lambda_noobj * F.mse_loss(noobj_predict_confidence, noobj_target_confidence, size_average=False)

        # 2. Compute the loss of containing object
        boxes_predict = coord_predict[:, :10]       # Match "delta_xy" in dataset.py
        boxes_target  = coord_target[:, :10]        # Match "delta_xy" in dataset.py
        
        N = boxes_predict.shape[0]                  # N: the number of bbox predicted
        # print("boxes_predict.shape: {}".format(boxes_predict.shape))
        # print("boxes_target.shape: {}".format(boxes_target.shape))
        boxes_predict_xy = torch.zeros_like(boxes_predict)
        boxes_target_xy  = torch.zeros_like(boxes_target)
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
        # print("IoU: {}".format(iou))
        # print("IoU.shape: {}".format(iou.shape))
        # print("Iou: {}".format(iou))
        iou_max, max_index = iou.max(dim=1)
        max_index = max_index.type(torch.ByteTensor)
        min_index = max_index.le(0)
        # print("IoU_Max.shape: {}".format(iou_max.shape))
        # print("IoU_Max: {}".format(iou_max))
        # print("max_index.shape: {}".format(max_index.shape))
        # print("max_index: {}".format(max_index))
        # print("max_index: {}".format(max_index.dtype))

        # Response Mask: the mask that notes the box need to calculate position loss.
        coord_response_mask = torch.zeros((iou_max.shape[0], 2), dtype=torch.uint8)
        coord_response_mask[max_index, 1] = 1
        coord_response_mask[min_index, 0] = 1
        # coord_response_mask = coord_response_mask.view(-1, 2)
        coord_response_mask = coord_response_mask.view(-1)
        coord_not_response_mask = coord_response_mask.le(0)
        # print("coord_response_mask.shape: {}".format(coord_response_mask.shape))
        # print("coord_response_mask: {}".format(coord_response_mask))
        # print("coord_not_response_mask.shape: {}".format(coord_not_response_mask.shape))
        # print("coord_not_response_mask: {}".format(coord_not_response_mask))

        boxes_predict = boxes_predict.contiguous().view(-1, 5)

        # Modify the Ground Truth
        # For 2.1 response loss: the gt of the confidence is the IoU(predict, target)
        # For 2.2 not response loss: the gt of the confidence is 0
        
        # boxes_target_iou = boxes_target.type(torch.cuda.FloatTensor).contiguous().view(-1, 5)
        boxes_target_iou = boxes_target.type(torch.float).contiguous().view(-1, 5)
        # print(boxes_target_iou.dtype)
        # print(iou_max.dtype)
        boxes_target_iou[coord_response_mask, 4]     = iou_max
        boxes_target_iou[coord_not_response_mask, 4] = 0
        # print("boxes_target_iou.shape: {}".format(boxes_target_iou.shape))
        # print("boxes_target_iou: {}".format(boxes_target_iou))

        boxes_predict_response = boxes_predict[coord_response_mask]
        boxes_target_response  = boxes_target_iou[coord_response_mask]
        # print("Boxes_predict_response.shape: {}".format(boxes_predict_response.shape))
        # print("Boxes_target_response.shape: {}".format(boxes_predict_response.shape))
        # print("Boxes_predict_response: {}".format(boxes_predict_response))
        # print("Boxes_target_response: {}".format(boxes_target_response))
        # boxes_target_response = boxes_target[coord_response_mask].view(-1, 5)

        # 2.1 response loss(Confidence, width & height, centerxy)
        # print("obj_confidence_loss: {}".format(F.mse_loss(boxes_predict_response[:, 4], boxes_target_response[:, 4], size_average=False)))
        # print("coord_xy_loss: {}".format(self.lambda_coord * F.mse_loss(boxes_predict_response[:, :2], boxes_target_response[:, :2], size_average=False)))
        # print("coord_hw_loss: {}".format(self.lambda_coord * F.mse_loss(torch.sqrt(boxes_predict_response[:, 2:4]), torch.sqrt(boxes_target_response[:, 2:4]), size_average=False)))
        loss += F.mse_loss(boxes_predict_response[:, 4], boxes_target_response[:, 4], size_average=False)
        loss += self.lambda_coord * F.mse_loss(boxes_predict_response[:, :2], boxes_target_response[:, :2], size_average=False)
        loss += self.lambda_coord * F.mse_loss(torch.sqrt(boxes_predict_response[:, 2:4]), torch.sqrt(boxes_target_response[:, 2:4]), size_average=False)

        # 2.2 not response loss, set the gt of the confidence as 0
        boxes_predict_not_response = boxes_predict[coord_not_response_mask]
        boxes_target_not_response  = boxes_target_iou[coord_not_response_mask]
        # boxes_target_not_response[:, 4] = 0
        # boxes_target_not_response[:, 9] = 0

        # print("noobj_confidence_loss: {}".format(self.lambda_noobj * F.mse_loss(boxes_predict_not_response[:, 4], boxes_target_not_response[:, 4], size_average=False)))
        loss += self.lambda_noobj * F.mse_loss(boxes_predict_not_response[:, 4], boxes_target_not_response[:, 4], size_average=False)
        # loss += self.lambda_noobj * F.mse_loss(boxes_predict_not_response[:, 9], boxes_target_not_response[:, 9], size_average=False)

        # 2.3 Compute the loss of class loss
        # print("class_loss: {}".format(F.mse_loss(coord_predict[:, 10:], coord_target[:, 10:], size_average=False)))
        coord_predict = coord_predict.type(torch.float)
        coord_target  = coord_target.type(torch.float)
        loss += F.mse_loss(coord_predict[:, 10:], coord_target[:, 10:], size_average=False)

        # Output the normalized loss
        loss /= batch_size

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
    device = utils.selectDevice(show=True)
    model = Yolov1_vgg16bn(pretrained=True).to(device)
    
    img    = torch.rand(1, 3, 448, 448).to(device)
    target = torch.rand(1, 7, 7, 26).to(device)
    output = model(img)
    
    # print(output.size())
    criterion = YoloLoss(7, 2, 5, 0.5, device)
    loss = criterion(output, target)

    print("Loss: {}".format(loss))

def loss_function_unittest():
    import math
    torch.set_default_dtype(torch.float)
    
    output = torch.zeros(1, 7, 7, 26)
    target = torch.zeros_like(output)

    obj   = torch.tensor([0.5, 0.5, 0.5, 0.5, 1])
    noobj = torch.tensor([0.5, 0.5, 0.5, 0.5, 0])
    classIndex = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float)
    target[:, 3, 3] = torch.cat((obj, obj, classIndex), dim=0)

    predict_obj = torch.tensor([0.5, 0.5, 1., 1., 0.5])
    classIndex = torch.tensor([0.9, 0.1, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    output[:, 3, 3] = torch.cat((predict_obj, predict_obj, classIndex), dim=0)

    criterion = YoloLoss(7, 2, 5, 0.5, "cpu")
    loss = criterion(output, target)

    print("*** loss_function_unittest: {}".format(loss.item()))
    print("*** loss_function_ref:")
    print("*** noobj_confidnce_loss: {}".format(0))
    print("*** obj_confidence_loss: {}".format((0.5 - 0.25) ** 2))
    print("*** coord_xy_loss: {}".format(0))
    print("*** coord_hw_loss: {}".format(5 * 2 * (math.sqrt(0.5) - math.sqrt(1)) ** 2))
    print("*** noobj_confidence_loss: {}".format(0.5 * (0.5 - 1) ** 2))
    print("*** class_loss: {}".format(2* 0.1 ** 2))
    
if __name__ == '__main__':
    # model_structure_unittest()
    
    loss_function_unittest()

    # model = Yolov1_vgg16bn(pretrained=True).to(device)
    # print(model)
