# IHM_instance_segmentation

to do:
- [ ] crop data around segmentation masks

Just do `bash setup.sh` or follow these instructions.

## Env

```bash
conda create -n ihm python=3.10 --y
conda activate ihm

pip install --no-cache-dir numpy==1.25.1
pip install --no-cache-dir torch==2.0.1
pip install --no-cache-dir transformers
pip install --no-cache-dir torchvision==0.15.2
pip install --no-cache-dir loadingpy
pip install --no-cache-dir opencv-python
pip install --no-cache-dir pycocotools
pip install --no-cache-dir torchinfo
```

download data from [drive](https://drive.google.com/file/d/1TrtT0WgVfiRZVGiOY8Zo2MmsGblq84z_/view?usp=sharing):
```bash
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1TrtT0WgVfiRZVGiOY8Zo2MmsGblq84z_' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1TrtT0WgVfiRZVGiOY8Zo2MmsGblq84z_" -O s.zip && rm -rf /tmp/cookies.txt
```

## How to use

Download COCO:
```bash
wget http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip
rm train2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip
rm annotations_trainval2017.zip
```

Generate the train, validation and test sets:
```bash
python -m src.main --generate
```
Once this is done, you will get two zipped folders, one for the students `s.zip` and one for you `t.zip`. The student folder contains the inputs and labels except for the test set. Your folder, contains the inputs, labels and visualization for all sets.

The second step would be to run the baseline in order to make sure that the problem can actually be solved.
```bash
python -m src.main --train
```

Then, you should gather all the students submissions in a folder (one sub-folder per student) and run
```bash
python -m src.main --evaluate
```

## Empirical validation of the hyper-parameters

in the wild (not available anymore)
| lr | steps | loss | mIoU |
| :---: | :---: | :---: | :---: |
| 1.0e-3 | 20000 | unbalanced | 15.8449 |
| 1.0e-3 | 50000 | balanced | 15.6928 |
| 1.0e-3 | 20000 | balanced | 15.8727 |
| 1.0e-3 | 10000 | balanced | 15.0999 |
| 5.0e-3 | 10000 | balanced | 13.3990 |
| 1.0e-3 | 7500 | balanced | 14.9102 |
| 5.0e-3 | 7500 | balanced | 14.1511 |
| 1.0e-3 | 5000 | balanced | 14.4383 |
| 5.0e-3 | 5000 | balanced | 13.8250 |
| 1.0e-3 | 2000 | balanced | 8.8968 |
| 5.0e-3 | 2000 | balanced | 13.4692 |

cropped images
| lr | steps | batch-size | mIoU |
| :---: | :---: | :---: | :---: |
| 1.0e-4 | 100,000 | 16 | 35.0018 |
| 1.0e-3 | 100,000 | 16 | 40.7762 |
| 5.0e-3 | 100,000 | 16 | 39.4309 |
| 1.0e-4 | 100,000 | 32 | 36.1988 |
| 1.0e-3 | 100,000 | 32 | 0.0084 |
| 5.0e-3 | 100,000 | 32 | 3.5473 |
| 1.0e-4 | 100,000 | 64 | 35.0714 |
| 1.0e-3 | 100,000 | 64 | 39.7016 |
| 5.0e-3 | 100,000 | 64 | 31.2726 |
| 1.0e-4 | 20,000 | 16 | 3.4551 |
| 1.0e-3 | 20,000 | 16 | 38.5191 |
| 5.0e-3 | 20,000 | 16 | 39.9602 |
| 1.0e-4 | 20,000 | 32 | 7.0918 |
| 1.0e-3 | 20,000 | 32 | 40.0492 |
| 5.0e-3 | 20,000 | 32 | 39.6928 |
| 1.0e-4 | 20,000 | 64 | 9.5360 |
| 1.0e-3 | 20,000 | 64 | 36.6459 |
| 5.0e-3 | 20,000 | 64 | 34.8041 |