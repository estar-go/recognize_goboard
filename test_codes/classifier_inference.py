import os, sys, torch, time, datetime, numpy as np, cv2, argparse, random
sys.path.append(os.path.split(sys.path[0])[0])
from configs.cls_eval_cfg import Config

from utils.server_utils import create_logger, B64Input, PIL2cv, cv2PIL, base64_to_img, get_image_transform
from models.backbone.cnn_utils import initialize_model
from utils.train_utils import load_checkpoint, tensor2np
from glob import glob
from tqdm import tqdm
from random import shuffle
import torch.nn as nn





if __name__ == '__main__':
    in_bpath = '/mnt/data/disk0/datasets/godatasets/goboard_datasets/test/'
    good_files = glob(in_bpath + 'sina/*/*.png')
    shuffle(good_files)
    good_files = good_files[:100]
    bad_files = glob(in_bpath + 'bad/*/*.jp*')
    shuffle(bad_files)
    bad_files = bad_files[:100]

    GLOBAL_CONFIG = Config()
    AI_MODEL = initialize_model(GLOBAL_CONFIG.cnn_type, GLOBAL_CONFIG.class_num)
    load_checkpoint(AI_MODEL, None, GLOBAL_CONFIG.weight_path)
    # AI_MODEL.cuda()
    AI_MODEL.eval()

    IMAGE_TRANSFORM = get_image_transform(GLOBAL_CONFIG.image_size)
    softmax = nn.Softmax()
    for now_file in tqdm(bad_files[:10] + good_files[:10]):
        img = cv2.imread(now_file)
        tensor_img = IMAGE_TRANSFORM(cv2PIL(img)).unsqueeze(0)
        with torch.set_grad_enabled(False):
            tensor_img = tensor_img.to(GLOBAL_CONFIG.device)
            output = AI_MODEL(tensor_img)
            result = softmax(output)
            print(result.cpu().detach().numpy().tolist())
            # prop, cls = torch.softmax(output, 1)
            # print(f"{prop}, {cls}")
        


