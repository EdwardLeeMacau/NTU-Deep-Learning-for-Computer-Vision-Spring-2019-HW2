import argparse

parser = argparse.ArgumentParser()

# Training
parser.add_argument("--worker", default=4, type=int)

# predict
parser.add_argument("--model", type=str, help="The model parameters file to read.")
parser.add_argument("--output", type=str, default="hw2_train_val/val1500/labelTxt_hbb_pred", help="The path to save the labels.")
parser.add_argument("--export", action="store_true", help="Store the results to the outputPath.")
args = parser.parse_args()
