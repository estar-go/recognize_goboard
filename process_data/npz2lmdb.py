from distutils import filelist
import os
import lmdb

import numpy as np
import argparse
import shutil
import glob
from tqdm import tqdm
import random

def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            if type(k) == str:
                k = k.encode()
            if type(v) == str:
                v = v.encode()
            txn.put(k, v)

def createDataset(out_train_path,
                  out_test_path,
                  fileList,
                  prop=0.95,
                  checkValid=False,
                  ):
    """
    Create LMDB dataset for CRNN training.
    ARGS:
        out_train_path    : train LMDB output path
        out_test_path : test LMDB output path
        fileList     : list of npz files
        prop   : train / all data split ratio, default=0.95
        checkValid    : if true, check the validity of npz
    """
    # If lmdb file already exists, remove it. Or the new data will add to it.
    if os.path.exists(out_train_path):
        print(f"删除原有的 {out_train_path}")
        shutil.rmtree(out_train_path)
        os.makedirs(out_train_path)
    else:
        os.makedirs(out_train_path)
    
    if os.path.exists(out_test_path):
        print(f"删除原有的 {out_test_path}")
        shutil.rmtree(out_test_path)
        os.makedirs(out_test_path)
    else:
        os.makedirs(out_test_path)

    n = len(fileList)
    train_env = lmdb.open(out_train_path, map_size=10995116277766)
    test_env = lmdb.open(out_test_path, map_size=10995116277766)
    train_cache = {}
    test_cache = {}
    train_cnt = 1
    test_cnt = 1
    for npz_file in tqdm(fileList, desc='load npz to tensor'):
        if checkValid:
            try:
                tmp = np.load(npz_file)
                feature = tmp['feature']
                label = tmp['label']
            except Exception as e:
                print(f'{npz_file} file demaged! pass')
                os.remove(npz_file)
                continue
        with np.load(npz_file) as npz:
            feature_npl = npz['feature']
            label_npl = npz['label'].tolist()
            # feature_npl = feature_npl.astype(np.uint8)
            n = len(label_npl)
            for i in tqdm(range(n)):
                is_train = True
                if random.random() >= prop:
                    is_train = False
                feature = feature_npl[i]
                label = label_npl[i]
                if is_train:
                    imageKey = f'image-{train_cnt :012d}'
                    labelKey = f'label-{train_cnt :012d}'
                    train_cache[imageKey] = feature
                    train_cache[labelKey] = str(label)
                    train_cnt += 1
                else:
                    imageKey = f'image-{test_cnt :012d}'
                    labelKey = f'label-{test_cnt :012d}'
                    test_cache[imageKey] = feature
                    test_cache[labelKey] = str(label)
                    test_cnt += 1
                
                if train_cnt % 10000 == 0:
                    train_samples = train_cnt - 1
                    train_cache['num-samples'] = str(train_samples)
                    writeCache(train_env, train_cache)
                    train_cache = {}
                    print('Written train %d ' % (train_cnt))
                    
                
                if test_cnt % 10000 == 0:
                    test_samples = test_cnt - 1
                    test_cache['num-samples'] = str(test_samples)
                    writeCache(test_env, test_cache)
                    test_cache = {}
                    print('Written test %d ' % (test_cnt))
                    
            
    train_samples = train_cnt - 1
    train_cache['num-samples'] = str(train_samples)
    writeCache(train_env, train_cache)
    train_env.close()
    print('Created train dataset with %d samples' % train_samples)
    
    test_samples = test_cnt - 1
    test_cache['num-samples'] = str(test_samples)
    writeCache(test_env, test_cache)
    test_env.close()
    print('Created test dataset with %d samples' % test_samples)


if __name__ == '__main__':
    # in_bpath = '/dev/shm/kgs_sgf_train_test_data/train/'
    in_bpath = '/data/datasets/kgs_all_npz/'
    # save_dir = '/dev/shm/kgs_all_lmdb'
    save_train_dir = '/data1/datasets/kgs_train_lmdb'
    save_test_dir = '/data1/datasets/kgs_test_lmdb'

    clean_bad_npz = False
    
    if clean_bad_npz:
        file_list = glob.glob(in_bpath + '*.npz')
        bad_cnt = 0
        for npz_file in tqdm(file_list, desc='cleanup npz...'):
            try:
                tmp = np.load(npz_file)
                features = tmp['feature']
                labels = tmp['label']
            except Exception as e:
                print(e)
                print(f'{npz_file} file demaged! delete')
                print(bad_cnt)
                os.remove(npz_file)
                bad_cnt += 1
                continue
            
        print(len(file_list), bad_cnt)

    # start create lmdb datasets
    in_file_list = glob.glob(in_bpath + '*.npz')
    createDataset(save_train_dir, save_test_dir, in_file_list)


    
