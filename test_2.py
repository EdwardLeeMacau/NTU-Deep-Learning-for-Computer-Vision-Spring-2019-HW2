import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import dataset
from torch.utils.data import Dataset, DataLoader
import utils

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

        # for j in range(7):
        #     for i in range(7):
        #         print("J: {}, I: {}".format(j, i))                
        #         print(coord_mask[0, j, i].data.tolist())
        #         print(noobj_mask[0, j, i].data.tolist())

        coord_predict = output[coord_mask].view(-1, 26).type(torch.float)
        coord_target  = target[coord_mask].view(-1, 26).type(torch.float)
        noobj_predict = output[noobj_mask].view(-1, 26).type(torch.float)
        noobj_target  = target[noobj_mask].view(-1, 26).type(torch.float)

        # for j in range(7):
            # for i in range(7):
                # print("J: {}, I: {}".format(j, i))                
        # print(coord_predict)
        # print(coord_target)

        """ Loss 1: Class_loss """
        class_loss = F.mse_loss(coord_predict[:, 10:], coord_target[:, 10:], size_average=False)
        print("*** class_loss: {}".format(class_loss.item()))

        """ Loss 2: No_object_Loss """
        no_object_loss = (F.mse_loss(noobj_predict[:, 4], noobj_target[:, 4], size_average=False)
                         + F.mse_loss(noobj_predict[:, 9], noobj_target[:, 9], size_average=False))
        print("*** no_object_loss: {}".format(no_object_loss.item()))

        # 2. Compute the loss of containing object
        boxes_predict = coord_predict[:, :10]       # Match "delta_xy" in dataset.py
        boxes_target  = coord_target[:, :10]        # Match "delta_xy" in dataset.py
        
        boxes_predict_xy = self.xyhw_xyxy(boxes_predict)
        boxes_target_xy = self.xyhw_xyxy(boxes_target)
        # print("*** boxes_predict_xy: {}".format(boxes_predict_xy))
        # print("*** boxes_target_xy: {}".format(boxes_target_xy))

        iou = self.IoU(boxes_predict_xy, boxes_target_xy)
        # print("*** IOU: {}".format(iou))
        # print("IoU: {}".format(iou))
        # print("IoU.shape: {}".format(iou.shape))
        # print("Iou: {}".format(iou))
        iou_max, max_index = iou.max(dim=1)
        # print("*** iou_max: {}".format(iou_max))
        # print("*** max_index: {}".format(max_index))
        max_index = max_index.type(torch.uint8)
        min_index = max_index.le(0)
        # print("*** max_index: {}".format(max_index))
        # print("*** min_index: {}".format(min_index))
        
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
        # print("*** response: {}".format(response_mask))
        # print("*** not_response: {}".format(not_response_mask))
        
        # print("coord_response_mask.shape: {}".format(coord_response_mask.shape))
        # print("coord_response_mask: {}".format(coord_response_mask))
        # print("coord_not_response_mask.shape: {}".format(coord_not_response_mask.shape))
        # print("coord_not_response_mask: {}".format(coord_not_response_mask))

        boxes_predict = boxes_predict.contiguous().view(-1, 5)
        boxes_target_iou = boxes_target.type(torch.float).contiguous().view(-1, 5)
        boxes_target_iou[response_mask, 4] = iou_max
        boxes_target_iou[not_response_mask, 4] = 0
        # print("*** box_predict: {}".format(boxes_predict))
        # print("*** box_target_iou: {}".format(boxes_target_iou))
        # print("boxes_target_iou.shape: {}".format(boxes_target_iou.shape))
        # print("boxes_target_iou: {}".format(boxes_target_iou))

        boxes_predict_response = boxes_predict[response_mask]
        boxes_target_response  = boxes_target_iou[response_mask]
        # print("*** box_predict_response: {}".format(boxes_predict_response))
        # print("*** box_target_response: {}".format(boxes_target_response))

        # print("Boxes_predict_response.shape: {}".format(boxes_predict_response.shape))
        # print("Boxes_target_response.shape: {}".format(boxes_predict_response.shape))
        # print("Boxes_predict_response: {}".format(boxes_predict_response))
        # print("Boxes_target_response: {}".format(boxes_target_response))
        # boxes_target_response = boxes_target[coord_response_mask].view(-1, 5)

        """ Class 3: Contain_loss """
        response_loss = F.mse_loss(boxes_predict_response[:, 4], boxes_target_response[:, 4], size_average=False)
        print("*** response_loss: {}".format(response_loss.item()))


        """ Class 4: Location_loss """
        location_loss = (F.mse_loss(boxes_predict_response[:, :2], boxes_target_response[:, :2], size_average=False) + 
                         F.mse_loss(torch.sqrt(boxes_predict_response[:, 2:4]), torch.sqrt(boxes_target_response[:, 2:4]), size_average=False))
        print("*** location_loss: {}".format(location_loss.item()))

        # 2.2 not response loss, set the gt of the confidence as 0
        boxes_predict_not_response = boxes_predict[not_response_mask]
        boxes_target_not_response  = boxes_target_iou[not_response_mask]
        # print("*** boxes_predict_not_response: {}".format(boxes_predict_not_response))
        # print("*** boxes_target_not_response: {}".format(boxes_target_not_response))

        # print("noobj_confidence_loss: {}".format(self.lambda_noobj * F.mse_loss(boxes_predict_not_response[:, 4], boxes_target_not_response[:, 4], size_average=False)))
        """ Class 5: Not_response_loss """
        not_response_loss = F.mse_loss(boxes_predict_not_response[:, 4], boxes_target_not_response[:, 4], size_average=False)
        print("*** not_response_loss: {}".format(not_response_loss.item()))


        # Output the normalized loss
        loss = self.lambda_coord * location_loss + class_loss + response_loss + self.lambda_noobj * (not_response_loss + no_object_loss)
        loss /= batch_size
        return loss

def gt_generator():
    trainset  = dataset.MyDataset(root="hw2_train_val/train15000", train=False, size=15000, transform=transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor()
    ]))

    trainset_loader = DataLoader(trainset, batch_size=1, shuffle=False, num_workers=0)
    
    for img, target, name in trainset_loader:
        return target, name

def loss_function_unittest():
    import math
    torch.set_default_dtype(torch.float)
    torch.manual_seed(1334)
    
    output = torch.zeros(1, 7, 7, 26)
    target = torch.zeros_like(output)

    # obj   = torch.tensor([0.5, 0.5, 0.5, 0.5, 1])
    # noobj = torch.tensor([0.5, 0.5, 0.5, 0.5, 0])
    # classIndex = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float)
    # target[:, 3, 3] = torch.cat((obj, obj, classIndex), dim=0)

    target, name = gt_generator()
    print(name)

    output = torch.rand(1, 7, 7, 26)
    # predict_obj = torch.tensor([0.5, 0.5, 1., 1., 0.5])
    # classIndex = torch.tensor([0.9, 0.1, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    # output[:, 3, 3] = torch.cat((predict_obj, predict_obj, classIndex), dim=0)

    func = lambda x: round(float(x), 4)
    for j in range(7):
        for i in range(7):
            # print("J: {}, I: {}".format(j, i))
            # print(list(map(func, target[:, j, i].data.tolist()[0])))
            # print(list(map(func, output[:, j, i].data.tolist()[0])))
            # print("")
            pass

    criterion = YoloLoss(7, 2, 5, 0.5, "cpu")
    loss = criterion(output, target)
    print(loss.item())
    # print("*** loss_function_unittest: {}".format(loss.item()))
    # print("*** loss_function_ref:")
    # print("*** noobj_confidnce_loss: {}".format(0))
    # print("*** obj_confidence_loss: {}".format((0.5 - 0.25) ** 2))
    # print("*** coord_xy_loss: {}".format(0))
    # print("*** coord_hw_loss: {}".format(5 * 2 * (math.sqrt(0.5) - math.sqrt(1)) ** 2))
    # print("*** noobj_confidence_loss: {}".format(0.5 * (0.5 - 1) ** 2))
    # print("*** class_loss: {}".format(2* 0.1 ** 2))

if __name__ == "__main__":
    # print(utils.labelEncoder.transform(utils.classnames))
    # raise NotImplementedError
    loss_function_unittest()