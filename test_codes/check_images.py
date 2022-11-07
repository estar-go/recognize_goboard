import os, sys, cv2, shutil
from tqdm import tqdm
from PIL import Image

from glob import glob


if __name__ == '__main__':
    in_bpath = '/mnt/data/disk0/datasets/godatasets/goboard_datasets/*/'
    bad_path = '/mnt/data/disk0/datasets/godatasets/goboard_datasets/bad_files/'
    if not os.path.exists(bad_path):
        os.makedirs(bad_path)
    file_list = glob(f"{in_bpath}*/*/*.jp*") + glob(f"{in_bpath}*/*/*.png")
    for now_file in tqdm(file_list[41700:]):
        try:
            image = Image.open(now_file).convert("RGB")
        except Exception as e:
            print(now_file)
            shutil.move(now_file, f"{bad_path}{os.path.split(now_file)[1]}")
