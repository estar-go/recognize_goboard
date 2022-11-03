import os, sys, numpy as np, random, torch, cv2
import torch.utils.data as data
from glob import glob

from tqdm import tqdm
import torchvision.transforms as transforms
from PIL import Image

class ImageFileDataset(data.Dataset):
    def __init__(self, base_data_path, transform, image_postfix='.png'):
        print('data init')
        self.base_data_path=base_data_path
        self.transform = transform
        self.image_postfix = image_postfix

        if os.path.isdir(base_data_path):
            self.fnames = self.get_data_list(base_data_path)
        elif os.path.isfile(base_data_path):
            print("暂不支持!")
            exit()        

    def get_data_list(self, base_data_path):
        file_list = glob(base_data_path + '*/*/*.schema')
        checked_file_list = []
        for now_label_file in tqdm(file_list, desc='check labels'):
            now_image_file = now_label_file.replace('.schema', self.image_postfix)
            if not os.path.exists(now_image_file):
                print(now_image_file + ' does not exist! pass...')
                continue
            checked_file_list.append(now_label_file)
        return checked_file_list

    def txt2np(self, in_file):
        with open(in_file, 'r') as f:
            s = f.readline().strip()[1:-1]
            if not s:
                label = np.zeros((3, 19, 19), np.float32)
                label[0, :, :] = 1.0
                return label
            lt = [int(i) for i in s.split(',')]
            n_l = np.array(lt, np.int32).reshape((19, 19)).T
            n_l[n_l==-1] = 2
            label = np.zeros((3, 19, 19), np.float32)
            label[0, n_l==0] = 1.0
            label[1, n_l==1] = 1.0
            label[2, n_l==2] = 1.0
            return label


    def __getitem__(self, idx):
        # 返回 图片 和对应的 3 x 19 x 19 numpy矩阵 
        now_label_file = self.fnames[idx]
        now_image_file = now_label_file.replace('.schema', self.image_postfix)
        
        img = Image.open(now_image_file).convert("RGB")
        img = self.transform(img)
        try:
            label = self.txt2np(now_label_file)
        except Exception as e:
            print(now_image_file)
            label = np.zeros((3, 19, 19), np.float32)
            label[0, :, :] = 1.0
        return img, torch.from_numpy(label)

    def __len__(self):
        return len(self.fnames)

if __name__ == '__main__':
    np.set_printoptions(threshold=sys.maxsize)
    transform = transforms.Compose([
        transforms.Resize((604, 604)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    datasets = ImageFileDataset(base_data_path='/data/datasets/godatasets/goboard_datasets/test/', transform=transform)
    print(len(datasets))
    it = 0
    for feature, label in datasets:
        # print(feature)
        # nl = label.numpy()
        # print(nl)
        # cv2.imshow('feature', feature)
        # if cv2.waitKey(1000000) & 0xFF == ord('q'):
        #     break
        # print(feature.shape, label.shape)
        pass
        
        it += 1
        if it > 3:
            break