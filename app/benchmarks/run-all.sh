#!/bin/bash

# Run the script from the directory where the script is located or adjust paths accordingly

BASE_MODEL=$1
RESULT_NAME=$2
ADAPTER_PATH="${3:-None}" # Optional, get the adapter from the /outputs directory

# # Arc-challange
python3 arc-chalange/arc-chalange.py --model_name_or_path $BASE_MODEL --result_name $RESULT_NAME --adapter_name $ADAPTER_PATH

# Arc-easy
python3 arc-easy/arc-easy.py --model_name_or_path $BASE_MODEL --result_name $RESULT_NAME --adapter_name $ADAPTER_PATH

# openbookqa
python3 openbookqa/openbookqa.py --model_name_or_path $BASE_MODEL --result_name $RESULT_NAME --adapter_name $ADAPTER_PATH

# siqa
python3 siqa/siqa.py --model_name_or_path $BASE_MODEL --result_name $RESULT_NAME --adapter_name $ADAPTER_PATH