# Adaptive Split-Fusion Transformer

### Update:
***19/02/2024***
- Release the pretrained model weights.

***31/08/2022***
- Add the code for ImageNet-21k pretrained weights loading.
- Add the code for the pyramid version model ASF-former_p.

***27/04/2022***
- The paper is posted on [arXiv](https://arxiv.org/abs/2204.12196) and the code is released.

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

ImageNet-1K Training:
|Model|Params|Top-1|
|-----|-----|-----|
|ASF-former-S|19.3M|[82.7%](https://drive.google.com/file/d/1Chx_Bi-pZJReP-WbB0loNxY_9tTWSVTb/view?usp=drive_link)|
|ASF-former-B|56.7M|[83.9%](https://drive.google.com/file/d/1se0gBrXRFUOAuYvHZhEqDWP4KPdwE0NL/view?usp=drive_link)|
|ASF-former_p-S|21.3M|[83.0%](https://drive.google.com/file/d/1o1nn3es1mAtYHlRGfuaaM4968_RqgeVa/view?usp=drive_link)|
|ASF-former_p-B|58.9M|[83.9%](https://drive.google.com/file/d/1yBA4r4-hB-F7-_UGGBjbrA2SKqKrirci/view?usp=drive_link)|

ImageNet-22K Pretraining + ImageNet-1K Fine-tuning:
|Model|Params|Top-1|
|-----|-----|-----|
|ASF-former-B|56.7M|[85.2%](https://drive.google.com/file/d/1xE3xFcYRY1ffczSA4LmJYc268GV1D4eV/view?usp=drive_link)|

## 3. Training

Train ASF-former-S with 8 GPUs:
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./distributed_train.sh 8 path/to/data --model ASF_former_S -b 64 --lr 5e-4 --weight-decay .05 --amp --img-size 224
```

Train ASF-former-B with 8 GPUs:
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./distributed_train.sh 8 path/to/data --model ASF_former_B -b 64 --lr 5e-4 --weight-decay .065 --amp --img-size 224
```

Train ASF-former_p-S with 8 GPUs:
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./distributed_train.sh 8 /data/imagenet --model ASF_former_p_S -b 64 --lr 5e-4 --weight-decay .05 --amp --img-size 224
```

Train ASF-former_p-B with 8 GPUs:
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./distributed_train.sh 8 /data/imagenet --model ASF_former_p_B -b 64 --lr 5e-4 --weight-decay .065 --amp --img-size 22
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

## Citation
```
@article{Su2022AdaptiveST,
  title={Adaptive Split-Fusion Transformer},
  author={Zixuan Su and Hao Zhang and Jingjing Chen and Lei Pang and Chong-Wah Ngo and Yu-Gang Jiang},
  journal={ArXiv},
  year={2022},
  volume={abs/2204.12196}
}
```

Our codes are based on [T2T-ViT](https://github.com/yitu-opensource/T2T-ViT).
