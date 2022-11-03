import os, shutil, random
from glob import glob
from tqdm import tqdm


if __name__ == '__main__':
    in_bpath = '/data/datasets/godatasets/goboard_datasets/'
    train_bpath = in_bpath + 'train/yehu/'
    test_bpath = in_bpath +'test/yehu/'
    train_prop = 0.7
    label_file_postfix = '.schema'
    image_file_postfix = '.png'

    label_file_list = glob(train_bpath + '*/*' + label_file_postfix)
    # print(label_file_list[:3])
    random.shuffle(label_file_list)
    # print(label_file_list[:3])
    n = len(label_file_list)
    print(f"total instance: {n}")
    train_n = int(n * train_prop)
    for now_label_file in tqdm(label_file_list[train_n:]):
        now_image_file = now_label_file.replace(label_file_postfix, image_file_postfix)
        now_out_base_path = os.path.split(now_label_file)[0].replace(train_bpath, test_bpath) + '/'
        if not os.path.exists(now_out_base_path):
            os.makedirs(now_out_base_path)
        now_file_id  = os.path.split(now_label_file)[1].split('.')[0]
        shutil.move(now_label_file, now_out_base_path + now_file_id + label_file_postfix)
        shutil.move(now_image_file, now_out_base_path + now_file_id + image_file_postfix)



    
