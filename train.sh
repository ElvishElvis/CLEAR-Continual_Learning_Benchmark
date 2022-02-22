YAML=$1
ROOT=./yaml
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train.py --yaml=./yaml/$YAML