import os
from random import shuffle
from tqdm import tqdm


if __name__ == '__main__':
    train_n = 38042726
    test_n = 6694731
    out_train_file = 'train_shuffle_index_list.txt'
    out_test_file = 'test_shuffle_index_list.txt'
    train_list = [i for i in range(1, train_n + 1)]
    test_list = [i for i in range(1, test_n + 1)]

    shuffle_n = 7
    print('shuffle train list')
    for i in tqdm(range(shuffle_n)):
        shuffle(train_list)
    print('shuffle test list')
    for i in tqdm(range(shuffle_n)):
        shuffle(test_list)

    with open(out_train_file, 'w') as f:
        for i in tqdm(train_list):
            f.write(f"{i}\n")
    
    with open(out_test_file, 'w') as f:
        for i in tqdm(test_list):
            f.write(f"{i}\n")