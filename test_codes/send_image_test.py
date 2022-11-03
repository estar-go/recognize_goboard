import requests
import base64, os, json, cv2, time, sys, numpy as np
from io import BytesIO

from PIL import Image
from glob import glob 
from tqdm import tqdm

def image_to_base64(img):
    output_buffer = BytesIO()
    img.save(output_buffer, format='JPEG')
    byte_data = output_buffer.getvalue()
    base64_str = base64.b64encode(byte_data)
    return base64_str

def img_to_base64(img):
    img = cv2.imencode('.jpg', img)[1]
    str_img = str(base64.b64encode(img))[2:-1]
    return str_img

def base64_to_image(base64_str):
    # base64_data = re.sub('^data:image/.+;base64,', '', base64_str)
    base64_data = base64_str
    byte_data = base64.b64decode(base64_data)
    image_data = BytesIO(byte_data)
    img = Image.open(image_data)
    return img

def getByte(path):
    with open(path, 'rb') as f:
        img_byte = base64.b64encode(f.read())
    img_str = img_byte.decode('ascii')
    return img_str

def send_base64_img(url, file_path):
    img = cv2.imread(file_path)
    b64_str = img_to_base64(img)
    # 拼接参数ELIFECYCLE
    data = {'image': b64_str}
    # 发送post请求到服务器端
    r = requests.post(url, data = json.dumps(data))
    return r.json()

if __name__ == "__main__":
    server_url = "http://0.0.0.0:8002/open-api/recogn_goboard"
    file_base_path = '/data/datasets/godatasets/goboard_datasets/test/yike/*/*.png'
    test_file_list = glob(file_base_path)


    np.printoptions(threshold=sys.maxsize)
    for now_img_file in tqdm(test_file_list):
        img = cv2.imread(now_img_file)
        cv2.imwrite('tmp.png', img)
        result = send_base64_img(server_url, now_img_file)
        print(np.array(result['matrix']))
        ctn = input('continue?')
        if ctn in ['q', 'n']:
            break




