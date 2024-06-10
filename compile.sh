#"""
#* Copyright (c) 2024 OPPO. All rights reserved.
#* Under license: MIT
#* For full license text, see LICENSE file in the repo root
#"""

#TORCH=$(python3 -c "import os; import torch; print(os.path.dirname(torch.__file__))")
#echo $TORCH
#exit

#######################
#How to run this file and save terminal output to a txt file:
# cd PROJECT_ROOT
# ./compile.sh 2>&1 | tee results/tmp_pt2dlc.txt
#######################

MY_PYTHON=python3
#MY_PYTHON=python3.10

cd third_party/maskrcnn_main

$MY_PYTHON setup.py clean
if [ -d build ]; then
  rm -rf build
fi

$MY_PYTHON setup.py build

cp -r build/lib* build/lib
