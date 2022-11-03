import os, sys, numpy as np
from kataboard import Board
from sgfmill import sgf, sgf_moves

from tqdm import tqdm
from glob import glob
from multiprocessing import Pool

def get_sensible_feature(feature, pyboard, player):
    for x in range(0, 19):
        for y in range(0, 19):
            feature[-3, x, y] = 1. if pyboard.would_be_legal(player, pyboard.loc(y, x)) else 0.

def fill_history_feature_16dims(feature, history_list):
    if len(history_list) <= 1:
        return
    offset = 4
    cnt = 0
    step_n = 16
    for color, move in history_list[::-1][1:step_n+1]:
        y, x = move
        feature[cnt + offset, y, x] = 1.0
        cnt += 1
            
def data_point_23dim(pyboard, board, color, move, history_list, board_size=19):
    # [black, white, empty, ones, history[:16], senibleness, zeros, player color] 共23维度
    board_array = np.zeros((23, board_size, board_size), dtype=np.float32)
    board_array[3] = 1.0
    if color == 'b':
        board_array[22] = 1.0
    for p in board.list_occupied_points():
        board_array[0 if p[0] == 'b' else 1, p[1][0], p[1][1]] = 1.0
    board_array[2,] = 1.0 - (board_array[0] + board_array[1])
    pycolor = 1 if color == 'b' else 2
    get_sensible_feature(board_array, pyboard, pycolor)
    fill_history_feature_16dims(board_array, history_list)
    return board_array, move[0]*board_size+move[1]

def save_npz(out_file, feature, label):
    if feature is None:
        print(f"{out_file} error!")
        return
    feature = np.packbits(feature.astype(np.uint8), axis=1)
    np.savez(out_file, feature=feature, label=label)

def process_one_sample(args_list):
    game_file, save_bpath = args_list

    
    pyboard = Board(size=19)
    with open(game_file) as f:
        contents = f.read().encode('ascii')
        game = sgf.Sgf_game.from_bytes(contents)
        board, plays = sgf_moves.get_setup_and_moves(game)
        play_history = []
        feature_l = None
        label_list = []        
        for color, move in plays:
            try:
                if move is None: continue
                row, col = move
                feature, label = data_point_23dim(pyboard, board, color, move, play_history)
                label_list.append(label)
                play_history.append([color, move])
                feature = np.expand_dims(feature, axis=0)
                if feature_l is None:
                    feature_l = feature
                else:
                    feature_l = np.append(feature_l, feature, axis=0)              
                pycolor = 1 if color == 'b' else 2
                board.play(row, col, color)
                pyboard.play(pycolor, pyboard.loc(col, row))
            except Exception as e:
                print(e)
                continue

    file_name = os.path.split(game_file)[1].split('.')[0]
    out_file = save_bpath + file_name + '.npz'
    save_npz(out_file, feature=feature_l, label=np.array(label_list, np.int32))

if __name__ == '__main__':
    
    np.set_printoptions(threshold=sys.maxsize)
    kgs_base_path = '/dev/shm/kgs_sgf_train_test_data/all/'
    save_base_path = '/data/datasets/formated/classgo/all_npz/'

    # kgs_base_path = 'datasets/KGS_sample/'
    # save_base_path = 'datasets/npz_sample/'
    if not os.path.exists(save_base_path):
        os.makedirs(save_base_path)

    fileList = glob(kgs_base_path + '*.sgf')
    n = len(fileList)
    worker_nums = 64
    args_list = [[fn, save_base_path] for fn in fileList]

    with Pool(worker_nums) as p:
        r = list(tqdm(p.imap(process_one_sample, args_list), total=n, desc='sgf to npz'))

        
        
        
            