"""
Count the parameters and FLOPs of model.
"""
import cv2
import torch
from models.asf_former import *
from utils import load_for_transfer_learning
from fvcore.nn import FlopCountAnalysis, flop_count_table, flop_count_str

if __name__ == '__main__':
    model = ASF_former_S()
    #load_for_transfer_learning(model, "81.7_T2T_ViTt_14.pth.tar", use_ema=True, strict=False, num_classes=1000)
    model.eval()

    input = cv2.imread("ILSVRC2012_val_00000293.JPEG")
    input = cv2.resize(input, (224, 224))
    input = torch.from_numpy(input).permute(2, 0, 1)
    input = input[None,:,:,:].float()

    flop = FlopCountAnalysis(model, input)
    print(flop_count_table(flop, max_depth=4))
    #print(flop_count_str(flop))
    #print(flop.total())