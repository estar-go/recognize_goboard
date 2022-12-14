#encoding:utf-8
import os, sys, time, numpy as np, cv2, copy, argparse, random, torch
from models.backbones.model_utils import weight_DP_to_single


if __name__ == "__main__":
    backbone_type = 'se-resnext50'
    backbone_type = 'mobilenet'
    backbone_type = 'pplcnetx05'
    in_weight_file = 'outputs/best_models/20211215_pplcnetx05_best.pth'
    out_weight_file = 'outputs/best_models/20211215_pplcnetx05_best_s.pth'

    backbone_type = 'pplcnetx025'
    in_weight_file = 'outputs/best_models/20211212_pplcnetx025_best.pth'
    out_weight_file = 'outputs/best_models/20211212_pplcnetx025_best_s.pth'

    backbone_type = 'pplcnetx05'
    in_weight_file = 'outputs/pplcnetx05_224_2022-04-20/best_models.pth'
    out_weight_file = 'outputs/best_models/20220420_pplcnetx05_best_cpu.pth'
    weight_DP_to_single(backbone_type, 4, in_weight_file, out_weight_file)