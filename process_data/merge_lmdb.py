import lmdb, numpy as np
from tqdm import tqdm
from random import shuffle

# 将两个lmdb文件合并成一个新的lmdb
def merge_lmdb(in_train_lmdb, in_test_lmdb, train_shuffle_index_list, test_shuffle_index_list, out_train_lmdb, out_test_lmdb):

    print('Merge start!')

    # env代表Environment, txn代表Transaction
    # 打开lmdb文件，读模式
    in_train_env = lmdb.open(in_train_lmdb, max_readers=5000)
    in_test_env = lmdb.open(in_test_lmdb, max_readers=5000)

    # 创建事务
    
    train_n = len(train_shuffle_index_list)
    test_n = len(test_shuffle_index_list)
    train_prop = 0.9
    train_sample_len = int(train_n * train_prop)
    test_sample_len = int(test_n * train_prop)
    print(train_n, train_sample_len, test_n, test_sample_len)
    print(train_shuffle_index_list[:5])
    print(test_shuffle_index_list[:5])

    # 打开lmdb文件，写模式，
    out_train_env = lmdb.open(out_train_lmdb, map_size=int(1e12))
    out_train_txn = out_train_env.begin(write=True)

    out_test_env = lmdb.open(out_test_lmdb, map_size=int(1e12))
    out_test_txn = out_test_env.begin(write=True)

    one_array = np.ones((1, 19, 19))
    print('start making test lmdb file!')
    with in_train_env.begin(write=False) as txn:
        cnt = 1
        for index in tqdm(train_shuffle_index_list[train_sample_len:], desc='make train-test lmdb'):
            label_key = f'label-{index :012d}'.encode()
            label = txn.get(label_key)
            img_key = f'image-{index :012d}'.encode()
            value = txn.get(img_key)
            value = np.fromstring(value, dtype=np.float32).reshape((3, 19, 19))
            value[2] = 1.0 - (value[0] + value[1])
            # value = np.concatenate((value, one_array))
            out_test_txn.put(f'label-{cnt :012d}'.encode(), label)
            out_test_txn.put(f'image-{cnt :012d}'.encode(), value)
            cnt += 1
            
            
            if(cnt % 10000 == 0):
                # 将数据写入数据库，必须的，否则数据不会写入到数据库中
                out_test_txn.commit()
                out_test_txn = out_test_env.begin(write=True)
        
        if(cnt % 10000 != 0):
            out_test_txn.commit()
            out_test_txn = out_test_env.begin(write=True)

    with in_test_env.begin(write=False) as txn:
        for index in tqdm(test_shuffle_index_list[test_sample_len:], desc='make test-test lmdb'):
            label_key = f'label-{index :012d}'.encode()
            label = txn.get(label_key)
            img_key = f'image-{index :012d}'.encode()
            value = txn.get(img_key)
            value = np.fromstring(value, dtype=np.float32).reshape((3, 19, 19))
            value[2] = 1.0 - (value[0] + value[1])
            # value = np.concatenate((value, one_array))
            out_test_txn.put(f'label-{cnt :012d}'.encode(), label)
            out_test_txn.put(f'image-{cnt :012d}'.encode(), value)
            cnt += 1
            if(cnt % 10000 == 0):
                # 将数据写入数据库，必须的，否则数据不会写入到数据库中
                out_test_txn.commit()
                out_test_txn = out_test_env.begin(write=True)

        test_len = cnt - 1
        print(f"test data len: {test_len}")
        out_test_txn.put('num-samples'.encode(), str(test_len).encode())
        print("commit len to lmdb")
        out_test_txn.commit()

    print('close test lmdb')
    out_test_env.close()

    print('start making train lmdb file!')
    # 遍历数据库
    
    with in_train_env.begin(write=False) as txn:
        cnt = 1
        for index in tqdm(train_shuffle_index_list[:train_sample_len], desc='make train-train lmdb'):
            label_key = f'label-{index :012d}'.encode()
            label = txn.get(label_key)
            img_key = f'image-{index :012d}'.encode()
            value = txn.get(img_key)
            value = np.fromstring(value, dtype=np.float32).reshape((3, 19, 19))
            value[2] = 1.0 - (value[0] + value[1])
            # value = np.concatenate((value, one_array))
            out_train_txn.put(f'label-{cnt :012d}'.encode(), label)
            out_train_txn.put(f'image-{cnt :012d}'.encode(), value)
            cnt += 1
            if(cnt % 100000 == 0):
                # 将数据写入数据库，必须的，否则数据不会写入到数据库中
                out_train_txn.commit()
                out_train_txn = out_train_env.begin(write=True)
        
        if(cnt % 100000 != 0):
            out_train_txn.commit()
            out_train_txn = out_train_env.begin(write=True)

    with in_test_env.begin(write=False) as txn:
        for index in tqdm(test_shuffle_index_list[:test_sample_len], desc='make test-train lmdb'):
            label_key = f'label-{index :012d}'.encode()
            label = txn.get(label_key)
            img_key = f'image-{index :012d}'.encode()
            value = txn.get(img_key)
            value = np.fromstring(value, dtype=np.float32).reshape((3, 19, 19))
            value[2] = 1.0 - (value[0] + value[1])
            # value = np.concatenate((value, one_array))
            out_train_txn.put(f'label-{cnt :012d}'.encode(), label)
            out_train_txn.put(f'image-{cnt :012d}'.encode(), value)
            cnt += 1
            if(cnt % 100000 == 0):
                # 将数据写入数据库，必须的，否则数据不会写入到数据库中
                out_train_txn.commit()
                out_train_txn = out_train_env.begin(write=True)
        
        train_len = cnt - 1
        print(f"train data len: {train_len}")
        # out_train_txn = out_train_env.begin(write=True)
        out_train_txn.put('num-samples'.encode(), str(train_len).encode())
        print("commit len to lmdb")
        out_train_txn.commit()
        # if(cnt % 1000 != 0):
        #     out_train_txn.commit()
        #     out_train_txn = out_train_env.begin(write=True)
    print('close train lmdb')
    out_train_env.close()

    

    # 关闭lmdb
    in_train_env.close()
    in_test_env.close()
    
    

    print('Merge success!')

    # 输出结果lmdb的状态信息，可以看到数据是否合并成功
    # print(out_train_env.stat())

def get_shuffle_index_list(in_file):
    l = []
    with open(in_file, 'r') as f:
        for line in f:
            l.append(int(line.strip()))
    return l

if __name__ == '__main__':
    in_train_lmdb = '/dev/shm/kgs_train'
    in_test_lmdb = '/dev/shm/kgs_test'

    # in_train_lmdb = '/data/datasets/formated/classgo/lmdb_folder/kgs_train'
    # in_test_lmdb = '/data/datasets/formated/classgo/lmdb_folder/kgs_test'

    merged_train_lmdb = '/data/datasets/kgs_shuffled_3dim_train'
    merged_test_lmdb = '/data/datasets/kgs_shuffled_3dim_test'

    train_shuffle_file = 'train_shuffle_index_list.txt'
    test_shuffle_file = 'test_shuffle_index_list.txt'
    print('read train shuffle index list')
    train_shuffle_index_list = get_shuffle_index_list(train_shuffle_file)
    print('read test shuffle index list')
    test_shuffle_index_list = get_shuffle_index_list(test_shuffle_file)
    # in_train_lmdb = '/data/datasets/recognition/lmdb_folder/202206_sync_part1'
    # in_test_lmdb = '/data/datasets/recognition/lmdb_folder/202206_sync_part2'
    # merged_lmdb = '/data/datasets/recognition/lmdb_folder/202206_sync_train'
    merge_lmdb(in_train_lmdb, in_test_lmdb, train_shuffle_index_list, test_shuffle_index_list, merged_train_lmdb, merged_test_lmdb)