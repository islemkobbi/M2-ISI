#!/bin/bash
echo "create conda environment ihm (few seconds)"
conda create -n ihm python=3.10 --y -q > tmp_stdout.txt
eval "$(conda shell.bash hook)"
conda activate ihm
rm tmp_stdout.txt

echo "install pip packages (few minutes)"
pip install -q --no-cache-dir numpy==1.25.1
pip install -q --no-cache-dir torch==2.0.1
pip install -q --no-cache-dir transformers
pip install -q --no-cache-dir torchvision==0.15.2
pip install -q --no-cache-dir loadingpy
pip install -q --no-cache-dir opencv-python
pip install -q --no-cache-dir pycocotools
pip install -q --no-cache-dir torchinfo

# echo "download and unzip data from COCO (several minutes)"
# wget -q http://images.cocodataset.org/zips/train2017.zip
# unzip -qq train2017.zip
# rm train2017.zip
# wget -q http://images.cocodataset.org/annotations/annotations_trainval2017.zip
# unzip -qq annotations_trainval2017.zip
# rm annotations_trainval2017.zip

python -m src.main --generate

rm *.npy
rm -r train2017
rm -r SegSets
rm -r annotations

python -m src.main --train
python -m src.main --evaluate