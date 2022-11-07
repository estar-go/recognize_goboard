import os, sys, torch, time, datetime, numpy as np, cv2, argparse, random
from configs.eval_cfg import Config
from utils.server_utils import create_logger, B64Input, PIL2cv, cv2PIL, base64_to_img, get_image_transform

from models.backbone.cnn_utils import initialize_model
from utils.train_utils import load_checkpoint, tensor2np
import torch.nn as nn
import logging, asyncio, aiohttp
import threading
from fastapi import FastAPI
from fastapi import File, Request, Form
from fastapi import UploadFile
from fastapi import Header

np.printoptions(threshold=sys.maxsize)

GLOBAL_CONFIG = Config()

SERVER_LOG = create_logger(GLOBAL_CONFIG.log_base_path, GLOBAL_CONFIG.log_name)

app = FastAPI()

AI_MODEL = initialize_model(GLOBAL_CONFIG.cnn_type, GLOBAL_CONFIG.class_num)
load_checkpoint(AI_MODEL, None, GLOBAL_CONFIG.weight_path)
if GLOBAL_CONFIG.device != 'cpu':
    AI_MODEL.cuda()
AI_MODEL.eval()
IMAGE_TRANSFORM = get_image_transform(GLOBAL_CONFIG.image_size)

CLS_MODEL = initialize_model(GLOBAL_CONFIG.cls_cnn_type, GLOBAL_CONFIG.cls_class_num)
load_checkpoint(CLS_MODEL, None, GLOBAL_CONFIG.cls_weight_path)
if GLOBAL_CONFIG.device != 'cpu':
    CLS_MODEL.cuda()
CLS_MODEL.eval()
CLS_IMAGE_TRANSFORM = get_image_transform(GLOBAL_CONFIG.cls_image_size)
SOFTMAX = nn.Softmax()

@app.post("/open-api/recogn_goboard")
def run_goboard_recogn(input_map: B64Input):
    img = base64_to_img(input_map.image)
    tensor_img = IMAGE_TRANSFORM(cv2PIL(img)).unsqueeze(0)
    with torch.set_grad_enabled(False):
        tensor_img = tensor_img.to(GLOBAL_CONFIG.device)
        output = AI_MODEL(tensor_img)
        pred_output = torch.argmax(output, dim=1)
        pred_output = tensor2np(pred_output)
        # SERVER_LOG.log(f"{pred_output}")
        result_map = {
            'matrix': pred_output.tolist()
        }

    return result_map

@app.post("/open-api/recogn_classifier")
def run_goboard_classifier(input_map: B64Input):
    img = base64_to_img(input_map.image)
    tensor_img = CLS_IMAGE_TRANSFORM(cv2PIL(img)).unsqueeze(0)
    with torch.set_grad_enabled(False):
        tensor_img = tensor_img.to(GLOBAL_CONFIG.device)
        output = CLS_MODEL(tensor_img)
        pred_output = SOFTMAX(output)
        # SERVER_LOG.log(f"{pred_output}")
        result_map = {
            'prop': pred_output.cpu().detach().numpy().tolist()[1]
        }

    return result_map

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app='server_main:app', host='0.0.0.0', port=int(GLOBAL_CONFIG.server_port), debug=False)#, workers=1)#, threaded=True)# , reload=True
    

