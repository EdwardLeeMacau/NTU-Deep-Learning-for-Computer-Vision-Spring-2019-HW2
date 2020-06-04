# TODO: create shell script for running your YoloV1-vgg16bn model

# Download the model
if [ ! -f "Yolov1-Improve.pth" ]; then
    wget -O Yolov1-Improve.pth https://www.dropbox.com/s/x4jb5zd3sxiu8rf/Yolov1-Improve.pth?dl=0
fi

# Run predict.py
# $1 Image directory
# $2 Detection directory
python predict.py --model Yolov1-Improve.pth --nms --images $1 --output $2 --iou 0.5 --prob 0.1 improve
# python visualize_bbox.py drawdet
# python hw2_evaluation_task.py hw2_train_val/val1500/labelTxt_hbb_pred_sh hw2_train_val/val1500/labelTxt_hbb
