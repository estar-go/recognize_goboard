from distutils import filelist
import os
import lmdb
import cv2
import numpy as np
import argparse
import shutil
import glob
from tqdm import tqdm
from sgfmill import sgf, sgf_moves

def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            if type(k) == str:
                k = k.encode()
            if type(v) == str:
                v = v.encode()
            txn.put(k, v)

def data_point_3dim(board, move, board_size=19):
    board_array = np.zeros((3, board_size, board_size), dtype=np.float32)
    # board_array[2] = 1.0
    for p in board.list_occupied_points():
        board_array[0 if p[0] == 'b' else 1, p[1][0], p[1][1]] = 1.0
    board_array[2,] = 1.0 - (board_array[0] + board_array[1])
    return board_array, move[0]*board_size+move[1]

def createDataset(outputPath,
                  fileList,
                  lexiconList=None,
                  checkValid=True):
    """
    Create LMDB dataset for CRNN training.
    ARGS:
        outputPath    : LMDB output path
        imagePathList : list of image path
        labelList     : list of corresponding groundtruth texts
        lexiconList   : (optional) list of lexicon lists
        checkValid    : if true, check the validity of every image
    """
    # If lmdb file already exists, remove it. Or the new data will add to it.
    if os.path.exists(outputPath):
        shutil.rmtree(outputPath)
        os.makedirs(outputPath)
    else:
        os.makedirs(outputPath)

    n = len(fileList)
    env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}
    cnt = 1
    for game_file in tqdm(fileList, desc='load sgf to tensor'):
        with open(game_file) as f:
            contents = f.read().encode('ascii')
            game = sgf.Sgf_game.from_bytes(contents)
            board, plays = sgf_moves.get_setup_and_moves(game)
            for color, move in plays:
                if move is None: continue
                row, col = move
                feature, label = data_point_3dim(board, move)
                board.play(row, col, color)
                imageKey = f'image-{cnt :012d}'
                labelKey = f'label-{cnt :012d}'
                cache[imageKey] = feature
                cache[labelKey] = str(label)
                if lexiconList:
                    lexiconKey = 'lexicon-%09d' % cnt
                    cache[lexiconKey] = ' '.join(lexiconList[i])
                if cnt % 1000 == 0:
                    writeCache(env, cache)
                    cache = {}
                    print('Written %d ' % (cnt))
                cnt += 1

    nSamples = cnt - 1
    cache['num-samples'] = str(nSamples)
    writeCache(env, cache)
    env.close()
    print('Created dataset with %d samples' % nSamples)


def read_data_from_folder(folder_path):
    image_path_list = []
    label_list = []
    pics = glob.glob(os.path.join(folder_path, '*.jpg'))
    for pic in pics:
        if not os.path.exists(pic.replace('.jpg', '.txt')):
            continue
        image_path_list.append(pic)
        with open(pic.replace('.jpg', '.txt'), 'r', encoding='utf-8') as fr:
            label = fr.readline().encode('utf-8').decode().strip('\n')
        label_list.append(label)
    return image_path_list, label_list


def read_data_from_file(file_path):
    image_path_list = []
    label_list = []
    with open(file_path, 'r', encoding='utf-8') as fr:
        for line in fr.readlines():
            line = line.encode('utf-8').decode().strip('\n')
            image_path, label = line.split('\t')

            if not os.path.exists(image_path):
                continue

            image_path_list.append(image_path)
            label_list.append(label)

    return image_path_list, label_list


def show_demo(demo_number, image_path_list, label_list):
    print('\nShow some demo to prevent creating wrong lmdb data')
    print(
        'The first line is the path to image and the second line is the image label'
    )
    for i in range(demo_number):
        print('image: %s\nlabel: %s\n' % (image_path_list[i], label_list[i]))


if __name__ == '__main__':
    in_bpath = '/dev/shm/kgs_sgf_train_test_data/train/'
    # save_dir = '/dev/shm/kgs_train'
    save_dir = '/data/datasets/formated/classgo/lmdb_folder/kgs_train'

    # in_bpath = '/dev/shm/kgs_sgf_train_test_data/test/'
    # save_dir = '/data/datasets/formated/classgo/lmdb_folder/kgs_test'
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    in_file_list = glob.glob(in_bpath + '*.sgf')
    createDataset(save_dir, in_file_list)


    # folder_flag = True
    # if not folder_flag:
    #     image_path_list, label_list = read_data_from_file(img_paths)
    #     createDataset(save_dir, image_path_list, label_list)
    #     show_demo(2, image_path_list, label_list)
    # else:
    #     image_path_list, label_list = read_data_from_folder(img_paths)
    #     createDataset(save_dir, image_path_list, label_list)
    #     show_demo(2, image_path_list, label_list)
