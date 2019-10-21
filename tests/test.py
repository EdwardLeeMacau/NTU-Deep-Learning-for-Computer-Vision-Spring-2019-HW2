import math

import utils
from dataset import MyDataset
from models import YoloLoss, Yolov1_vgg16bn_Improve
from predict import decode, labelEncoder

def augment_unittest():
    """ From dataset.py """

    trainset = MyDataset(root="hw2_train_val/train15000", train=True, transform=transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]))
    
    train_loader = DataLoader(trainset, batch_size=1, shuffle=False, num_workers=0)

    for _, target, _ in train_loader:
        boxes, classIndexs, probs = decode(target, nms=True, prob_min=0.1, iou_threshold=0.5)
        classNames = labelEncoder.inverse_transform(classIndexs.type(torch.long))

        boxes = (boxes * 512).round()
        rect  = torch.zeros(boxes.shape[0], 8)

        # Extand (x1, y1, x2, y2) to (x1, y1, x2, y1, x2, y2, x1, y2)
        rect[:,  :3] = boxes[:, :3]
        rect[:, 3:6] = boxes[:, 1:]
        rect[:, 6]   = boxes[:, 0]
        rect[:, 7]   = boxes[:, 3]

        # Return the probs to string lists
        round_func = lambda x: round(x, 3)
        probs = list(map(str, list(map(round_func, probs.data.tolist()))))
        classNames = list(map(str, classNames))

        for i in range(0, rect.shape[0]):
            prob = probs[i]
            className = classNames[i]

            print(rect[i], className, prob)

def loss_function_unittest():
    """ From model.py """
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

def model_structure_unittest():
    """ From model.py """
    device = utils.selectDevice(show=True)
    model = Yolov1_vgg16bn_Improve(pretrained=True).to(device)

    print(model)
    
    img    = torch.rand(1, 3, 448, 448).to(device)
    target = torch.rand(1, 14, 14, 26).to(device)
    output = model(img)
    
    # print(output.size())
    criterion = YoloLoss(14, 2, 5, 0.5, device)
    loss = criterion(output, target)

    print("Loss: {}".format(loss))

def decode_unittest():
    """ From predict.py """
    output = torch.zeros(1, 7, 7, 26)
    target = torch.zeros_like(output)

    obj   = torch.tensor([0.5, 0.5, 0.2, 0.8, 1])
    classIndex = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float)
    target[:, 3, 3] = torch.cat((obj, obj, classIndex), dim=0)
    output[:, 3, 3] = torch.cat((obj, obj, classIndex), dim=0)

    boxes, classIndexs, probs = decode(output, prob_min=0.1, iou_threshold=0.5, grid_num=7, bbox_num=2)
    classNames = labelEncoder.inverse_transform(classIndexs.type(torch.long).to("cpu"))

def encoder_unittest():
    """ From predict.py """
    print("*** classnames: \n{}".format(classnames))
    
    indexs = labelEncoder.transform(classnames).reshape(-1, 1)
    print("*** indexs: \n{}".format(indexs))
    
    onehot = oneHotEncoder.transform(indexs)
    print("*** onehot: \n{}".format(onehot))
    
    reverse_index = oneHotEncoder.inverse_transform(onehot).reshape(-1)
    print("*** reverse index: \n{}".format(reverse_index))
    
    reverse_classnames = labelEncoder.inverse_transform(reverse_index.astype(int))
    print("*** reverse classnames: \n{}".format(reverse_classnames))
    
def system_unittest():
    """ From predict.py """
    dataset  = dataset.MyDataset(root="hw2_train_val/train15000", train=False, size=15000, transform=transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor()
    ]))

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    
    # Testset prediction
    for _, target, labelName in loader:
        boxes, classIndexs, probs = decode(target, prob_min=0, iou_threshold=0.5)
        classNames = labelEncoder.inverse_transform(classIndexs.type(torch.long).to("cpu"))
        
        print("Raw Data: ")
        with open(labelName[0], "r") as textfile:
            content = textfile.readlines()
            print("\n".join(content))

        print("My Decoder: ")
        boxes = (boxes * 512).round()
        rect  = torch.zeros(boxes.shape[0], 8)
        rect[:,  :3] = boxes[:, :3]
        rect[:, 3:6] = boxes[:, 1:]
        rect[:, 6]   = boxes[:, 0]
        rect[:, 7]   = boxes[:, 3]
        round_func = lambda x: round(x, 3)
        probs = list(map(str, list(map(round_func, probs.data.tolist()))))
        classNames = list(map(str, classNames))
        for i in range(0, rect.shape[0]):
            prob = probs[i]
            className = classNames[i]
            print(" ".join(map(str, rect[i].data.tolist())), end=" ")
            print(" ".join((className, prob)))

        pdb.set_trace()
