# TODO: create shell script for running your YoloV1-vgg16bn model

# Download the model
# wget -O Yolov1-Improve.pth 

# Run predict.py
# $1 Image directory
# $2 Detection directory
python predict.py --model $1 --images hw2_train_val/val1500/images --output hw2_train_val/val1500/labelTxt_hbb_pred
python visualize_boox.py drawdet
python hw2_evaluation_task.py hw2_train_val/val1500/labelTxt_hbb_pred hw2_train_val/val1500/labelTxt_hbb