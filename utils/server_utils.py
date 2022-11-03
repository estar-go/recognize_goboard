import os, numpy as np, random, cv2, logging, json, torch
from rich.logging import RichHandler
from rich import print 

from time import strftime, gmtime
from pydantic import BaseModel
from typing import Optional
import urllib, cv2, numpy as np, base64, io
from PIL import Image
import torchvision.transforms as transforms

class B64Input(BaseModel):
    image: str

def img_to_base64(img):
    img = cv2.imencode('.jpg', img)[1]
    str_img = str(base64.b64encode(img))[2:-1]
    return str_img

def base64_to_img(base64_str):
    str_img = base64.b64decode(base64_str)
    image = Image.open(io.BytesIO(str_img))
    return cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)

def PIL2cv(image):
    return cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)

def cv2PIL(img):
    return Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))

def create_logger(base_path, log_name):
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    
    FORMAT = "%(message)s"
    logging.basicConfig(
        level="NOTSET",
        format=FORMAT,
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, tracebacks_show_locals=True, markup=True)]
    )

    logger = logging.getLogger(log_name)
    logger.setLevel(logging.NOTSET)

    fhander = logging.FileHandler('%s/%s'%(base_path, log_name))
    fhander.setLevel(logging.NOTSET)

    logger.addHandler(fhander)
    return logger

def get_image_transform(image_size=608):
    image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    return image_transform
