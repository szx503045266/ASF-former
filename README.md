# Adaptive Split-Fusion Transformer

### Update:
***27/04/2022***
- The code is released.

## 1. Requirements
timm==0.3.4

torch==1.8.0

torchvision

pyyaml

### Data Preparation
[ImageNet](https://image-net.org/) with following folder structure:

```
│imagenet/
├──train/
│  ├── n01440764
│  │   ├── n01440764_10026.JPEG
│  │   ├── n01440764_10027.JPEG
│  │   ├── ......
│  ├── ......
├──val/
│  ├── n01440764
│  │   ├── ILSVRC2012_val_00000293.JPEG
│  │   ├── ILSVRC2012_val_00002138.JPEG
│  │   ├── ......
│  ├── ......
```

## 2. Pretrained Models

The pretrained models will be released soon.

## 3. Training

Train ASF-former-S with 8 GPUs:
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./distributed_train.sh 8 path/to/data --model ASF_former_S -b 64 --lr 5e-4 --weight-decay .05 --amp --img-size 224
```

Train ASF-former-B with 8 GPUs:
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./distributed_train.sh 8 path/to/data --model ASF_former_B -b 64 --lr 5e-4 --weight-decay .065 --amp --img-size 224
```

## 4. Evaluation

Evaluate the ASF-former-S model:
```
CUDA_VISIBLE_DEVICES=0 python main.py path/to/data --model ASF_former_S -b 100 --eval_checkpoint path/to/checkpoint
```

Evaluate the ASF-former-B model:
```
CUDA_VISIBLE_DEVICES=0 python main.py path/to/data --model ASF_former_B -b 100 --eval_checkpoint path/to/checkpoint
```

## 5. Transfer Learning

Transfer ASF-former-S to CIFAR-10:
```
CUDA_VISIBLE_DEVICES=0,1 python transfer_learning.py --lr 0.025 --b 64 --num-classes 10 --img-size 224 --transfer-learning True --transfer-model path/to/model
```
Transfer ASF-former-B to CIFAR-10:
```
CUDA_VISIBLE_DEVICES=0,1 python transfer_learning.py --lr 0.025 --b 64 --num-classes 10 --img-size 224 --transfer-learning True --transfer-model path/to/model --model ASF_former_B
```
Transfer ASF-former-S to CIFAR-100:
```
CUDA_VISIBLE_DEVICES=0,1 python transfer_learning.py --lr 0.05 --b 64 --num-classes 100 --img-size 224 --transfer-learning True --transfer-model path/to/model --dataset cifar100
```
Transfer ASF-former-B to CIFAR-100:
```
CUDA_VISIBLE_DEVICES=0,1 python transfer_learning.py --lr 0.05 --b 64 --num-classes 100 --img-size 224 --transfer-learning True --transfer-model path/to/model --dataset cifar100 --model ASF_former_B
```

Our codes are based on [T2T-ViT](https://github.com/yitu-opensource/T2T-ViT).
