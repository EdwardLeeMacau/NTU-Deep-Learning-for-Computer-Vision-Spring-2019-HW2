# TODO: create shell script for running your YoloV1-vgg16bn model

# Download the model
if [ ! -f "Yolov1.pth" ]; then
    wget -O Yolov1.pth https://www.dropbox.com/s/2z3xjy2ovdeli41/Yolov1.pth?dl=0
fi

# Run predict.py
# $1 Image directory
# $2 Detection directory
python predict.py --model Yolov1.pth --images $1 --output $2 --nms --iou 0.6 --prob 0.05 basic
# python visualize_boox.py drawdet
# python hw2_evaluation_task.py hw2_train_val/val1500/labelTxt_hbb_pred hw2_train_val/val1500/labelTxt_hbb