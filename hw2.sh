# TODO: create shell script for running your YoloV1-vgg16bn model

# Download the model
wget -O Yolov1.pth 

# Run predict.py
python predict.py val --model Yolov1.pth

# Run Evaluation_task.py
python hw2_evaluation_task.py hw2_train_val/val1500/labelTxt_hbb_pred/ hw2_train_val/val1500/labelTxt_hbb/