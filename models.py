import torch.nn as nn
import math
import torch
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import utils
from torch.autograd import Variable

__all__ = ['vgg16_bn']
model_urls = {'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',}

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
            # nn.BatchNorm1d(num_features=4096),
            nn.LeakyReLU(negative_slope=0.02),
            nn.Dropout(0.5),

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
class YoloLoss_github(nn.Module):
    def __init__(self, S, B, l_coord, l_noobj):
        super(YoloLoss_github, self).__init__()
        self.S = S
        self.B = B
        self.l_coord = l_coord
        self.l_noobj = l_noobj

    def compute_iou(self, box1, box2):
        '''Compute the intersection over union of two set of boxes, each box is [x1,y1,x2,y2].
        Args:
          box1: (tensor) bounding boxes, sized [N,4].
          box2: (tensor) bounding boxes, sized [M,4].
        Return:
          (tensor) iou, sized [N,M].
        '''
        N = box1.size(0)
        M = box2.size(0)

        lt = torch.max(
            box1[:,:2].unsqueeze(1).expand(N,M,2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:,:2].unsqueeze(0).expand(N,M,2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )

        rb = torch.min(
            box1[:,2:].unsqueeze(1).expand(N,M,2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:,2:].unsqueeze(0).expand(N,M,2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )

        wh = rb - lt  # [N,M,2]
        wh[wh<0] = 0  # clip at 0
        inter = wh[:,:,0] * wh[:,:,1]  # [N,M]

        area1 = (box1[:,2]-box1[:,0]) * (box1[:,3]-box1[:,1])  # [N,]
        area2 = (box2[:,2]-box2[:,0]) * (box2[:,3]-box2[:,1])  # [M,]
        area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
        area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

        iou = inter / (area1 + area2 - inter)
        return iou

    def forward(self, predict, target):
        '''
        pred_tensor: (tensor) size(batchsize,S,S,Bx5+20=30) [x,y,w,h,c]
        target_tensor: (tensor) size(batchsize,S,S,30)
        '''
        N = predict.size()[0]
        coo_mask = (target[:,:,:,4] > 0).unsqueeze(-1).expand_as(target)
        noo_mask = (target[:,:,:,4] == 0).unsqueeze(-1).expand_as(target)

        coo_pred = predict[coo_mask].view(-1,26)
        box_pred = coo_pred[:,:10].contiguous().view(-1,5) #box[x1,y1,w1,h1,c1]
        class_pred = coo_pred[:,10:]                       #[x2,y2,w2,h2,c2]
        
        coo_target = target[coo_mask].view(-1,26)
        box_target = coo_target[:,:10].contiguous().view(-1,5)
        class_target = coo_target[:,10:]

        # compute not contain obj loss
        noo_pred = predict[noo_mask].view(-1,26)
        noo_target = target[noo_mask].view(-1,26)
        noo_pred_mask = torch.ByteTensor(noo_pred.size())
        noo_pred_mask.zero_()
        noo_pred_mask[:,4] = 1
        noo_pred_mask[:,9] = 1
        noo_pred_c = noo_pred[noo_pred_mask] #noo pred只需要计算 c 的损失 size[-1,2]
        noo_target_c = noo_target[noo_pred_mask]
        nooobj_loss = F.mse_loss(noo_pred_c,noo_target_c,size_average=False)

        # compute contain obj loss
        coo_response_mask = torch.ByteTensor(box_target.size())
        coo_response_mask.zero_()
        coo_not_response_mask = torch.ByteTensor(box_target.size())
        coo_not_response_mask.zero_()
        box_target_iou = torch.zeros(box_target.size())
        
        for i in range(0, box_target.size()[0], 2): #choose the best iou box
            box1 = box_pred[i:i+2]
            box1_xyxy = Variable(torch.FloatTensor(box1.size()))
            box1_xyxy[:,:2] = box1[:,:2] / 7. - 0.5 * box1[:,2:4]
            box1_xyxy[:,2:4] = box1[:,:2] / 7. + 0.5 * box1[:,2:4]
            box2 = box_target[i].view(-1,5)
            box2_xyxy = Variable(torch.FloatTensor(box2.size()))
            box2_xyxy[:,:2] = box2[:,:2] / 7. - 0.5 * box2[:,2:4]
            box2_xyxy[:,2:4] = box2[:,:2] / 7. + 0.5 * box2[:,2:4]
            iou = self.compute_iou(box1_xyxy[:,:4],box2_xyxy[:,:4]) #[2,1]
            max_iou,max_index = iou.max(0)
            max_index = max_index.data
            
            coo_response_mask[i+max_index]=1
            coo_not_response_mask[i+1-max_index]=1

            box_target_iou[i+max_index,torch.LongTensor([4])] = (max_iou).data

        box_target_iou = Variable(box_target_iou)
        #1.response loss
        box_pred_response = box_pred[coo_response_mask].view(-1,5)
        box_target_response_iou = box_target_iou[coo_response_mask].view(-1,5)
        box_target_response = box_target[coo_response_mask].view(-1,5)
        contain_loss = F.mse_loss(box_pred_response[:,4],box_target_response_iou[:,4],size_average=False)
        loc_loss = (F.mse_loss(box_pred_response[:,:2],box_target_response[:,:2],size_average=False) + 
                    F.mse_loss(torch.sqrt(box_pred_response[:,2:4]),torch.sqrt(box_target_response[:,2:4]),size_average=False))

        #2.not response loss
        box_pred_not_response = box_pred[coo_not_response_mask].view(-1,5)
        box_target_not_response = box_target[coo_not_response_mask].view(-1,5)
        box_target_not_response[:, 4] = 0
        #not_contain_loss = F.mse_loss(box_pred_response[:,4],box_target_response[:,4],size_average=False)
        
        #I believe this bug is simply a typo
        not_contain_loss = F.mse_loss(box_pred_not_response[:,4], box_target_not_response[:,4],size_average=False)

        #3.class loss
        class_loss = F.mse_loss(class_pred,class_target,size_average=False)

        # print("Class_loss: {}".format(class_loss))
        # print("No_object_loss: {}".format(nooobj_loss.item()))
        # print("Response_loss: {}".format(contain_loss.item()))
        # print("Location_loss: {}".format(loc_loss.item()))
        # print("Not_response_loss: {}".format(not_contain_loss.item()))

        # return (self.l_coord*loc_loss + 2*contain_loss + not_contain_loss + self.l_noobj*nooobj_loss + class_loss)/N
        return (self.l_coord * loc_loss + contain_loss + not_contain_loss + self.l_noobj * nooobj_loss + class_loss) / N

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

    def xywh_xyxy(self, tensor: torch.tensor):
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
        
        boxes_predict_xy = self.xywh_xyxy(boxes_predict)
        boxes_target_xy = self.xywh_xyxy(boxes_target)
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

class VGG_improve(nn.Module):
    """
      input_size  = 448 * 448
      output_size = 14 * 14 * (5 * 2 + 16) = 1274
    """
    def __init__(self, features, output_size=1274, image_size=448):
        super(VGG_improve, self).__init__()
        self.features = features
        self.image_size = image_size

        self.yolo = nn.Sequential(
            nn.Linear(512 * 14 * 14, 8092),
            nn.BatchNorm1d(num_features=8092),
            nn.LeakyReLU(negative_slope=0.02),
            nn.Dropout(0.5),

            nn.Linear(8092, 14 * 14 * 26)
        )
        self._initialize_weights()

    def forward(self, x):
        """
          input_size:    n * 3 * 448 * 448
          VGG16_bn:      n * 512 * 14 * 14
          Flatten Layer: n * 
          Yolo Layer:    n * 
          Sigmoid Layer: n * 
          Reshape Layer: n * 14 * 14 * 26
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
        
        x = x.view(-1, 14, 14, 26)
        # print(x.shape, x.dtype)
        
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def make_layers_improve(cfg, batch_norm=False):
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

    for v in cfg:
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

def Yolov1_vgg16bn_improve(pretrained=False, **kwargs):
    """
    VGG 16-layer model (configuration "D") with batch normalization
    
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
            
    Return:
        yolo: the prediction model YOLO.
    """

    # print(make_layers(cfg['D'], batch_norm=True))

    yolo = VGG_improve(make_layers_improve(cfg['D'], batch_norm=True), **kwargs)

    vgg_state_dict = model_zoo.load_url(model_urls['vgg16_bn'])
    yolo_state_dict = yolo.state_dict()
    for k in vgg_state_dict.keys():
        if k in yolo_state_dict.keys() and k.startswith('features'):
            yolo_state_dict[k] = vgg_state_dict[k]
    
    yolo.load_state_dict(yolo_state_dict)
    
    return yolo

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
    model = Yolov1_vgg16bn_improve(pretrained=True).to(device)

    print(model)
    
    img    = torch.rand(1, 3, 448, 448).to(device)
    target = torch.rand(1, 14, 14, 26).to(device)
    output = model(img)
    
    # print(output.size())
    criterion = YoloLoss(14, 2, 5, 0.5, device)
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
    model_structure_unittest()
    
    # loss_function_unittest()

    # model = Yolov1_vgg16bn(pretrained=True).to(device)
    # print(model)
