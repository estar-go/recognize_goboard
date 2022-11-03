import os, numpy as np, sys
from glob import glob
from tqdm import tqdm


if __name__ == '__main__':
    np.printoptions(threshold=sys.maxsize)
    fileList = glob('datasets/npz_sample/*.npz')

    for npz_file in tqdm(fileList, desc='load npz to tensor'):
            
            with np.load(npz_file) as npz:
                feature_npl = npz['feature']
                label_npl = npz['label'].tolist()
                n = len(label_npl)
                for i in range(n):
                    feature = feature_npl[i]
                    label = label_npl[i]
                    feature = np.unpackbits(feature, axis=0, count=-1).astype(np.float32)
                    # shape : 23, 19, 19
                    print('黑')
                    print(feature[0])
                    c = input("continue? ")
                    if c == 'q':
                        exit()
                    print("白")
                    print(feature[1])
                    c = input("continue? ")
                    if c == 'q':
                        exit()
                    print("空")
                    print(feature[2])
                    c = input("continue? ")
                    if c == 'q':
                        exit()
                    # print("1")
                    # print(feature[3])
                    # c = input("continue? ")
                    # if c == 'q':
                    #     exit()
                    print("h0")
                    print(feature[4])
                    c = input("continue? ")
                    if c == 'q':
                        exit()
                        
                    print("h1")
                    print(feature[5])
                    c = input("continue? ")
                    if c == 'q':
                        exit()

                    print("h2")
                    print(feature[6])
                    c = input("continue? ")
                    if c == 'q':
                        exit()

                    print("h3")
                    print(feature[7])
                    c = input("continue? ")
                    if c == 'q':
                        exit()

                    print("sensible")
                    print(feature[-3])
                    c = input("continue? ")
                    if c == 'q':
                        exit()

                    print("label")
                    print(label)
                    print('-'*50)
                    c = input("continue? ")
                    if c == 'q':
                        exit()