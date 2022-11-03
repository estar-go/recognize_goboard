import numpy as np
import sys


# 读取保存的numpy数组
def read_npy(in_file):
    return np.load(in_file)

# 写入当前盘面的label数组, 要保证写入的文件名和图片一一对应: 1.png 对应 1.npy
def write_npy(np_label, out_file):
    np.save(out_file, np_label)


if __name__ == '__main__':
    np.set_printoptions(threshold=sys.maxsize)
    base_path = 'goboard_data_demo/'
    # 空矩阵
    np_board = np.zeros((19, 19), np.uint8)
    # 下棋历史 points_list：
    # []为空棋盘无人下棋 
    # 矩阵初始化为全0矩阵，1代表黑子，2代表白子
    # [14, 14, 1]表示黑子落在(14, 14)的位置, 矩阵坐标边界为[0, 18]
    points_list = [
        [], 
        [15, 15, 1],
        [2, 15, 2],
        [3, 15, 1],
        [3, 16, 2],
    ]
    for it, lt in enumerate(points_list):
        out_label_file = f"{base_path}{it}.npy"
        if lt == []:
            pass
        else:
            x, y, color = lt
            # 注意：这里是 y, x, color，要把x和y倒过来
            np_board[y, x] = color

        # 将当前盘面点位结果保存
        write_npy(np_board, out_label_file)
        # 读取保存的盘面数组
        show_board = read_npy(out_label_file)
        print(f"{it}:")
        # 打印读取的和写入的label是否一致
        print((show_board == np_board).all())
        # 打印当前盘面
        print(show_board)

            






