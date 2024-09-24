# *_*coding:utf-8 *_*
"""
cmd:
cd preprocess
python mafw.py
"""

import os
import pandas as pd
import sys

# change 'data_path' to yours
dataset = "MAFW"
data_path = os.path.expanduser(f'~/AC/Dataset/{dataset}')
split_dir = os.path.join(data_path, 'Train & Test Set/single/no_caption')
video_dir = os.path.join(data_path, 'data/frames')


num_splits = 5
for split in range(1, 1 + num_splits):
    save_dir = f'../saved/data/mafw/single/split0{split}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    labels = ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise', 'contempt', 'anxiety', 'helplessness', 'disappointment']
    label2idx = {l:idx for idx, l in enumerate(labels)}

    # read
    train_split_file = os.path.join(split_dir, f'set_{split}/train.txt')
    df = pd.read_csv(train_split_file, header=None, delimiter=' ')
    train_label_dict = dict(zip(df[0], df[1]))

    test_split_file = os.path.join(split_dir, f'set_{split}/test.txt')
    df = pd.read_csv(test_split_file, header=None, delimiter=' ')
    test_label_dict = dict(zip(df[0], df[1]))

    train_label_dict = {os.path.join(video_dir, f"{v.split('.')[0]}"):label2idx[l] for v,l in train_label_dict.items()}
    test_label_dict = {os.path.join(video_dir,  f"{v.split('.')[0]}"):label2idx[l] for v,l in test_label_dict.items()}

    total_samples = len(train_label_dict) + len(test_label_dict)
    print(f'Total samples in split {split}: {total_samples}, train={len(train_label_dict)}, test={len(test_label_dict)}')

    # write
    new_train_split_file = os.path.join(save_dir, f'train.csv')
    df = pd.DataFrame(train_label_dict.items())
    df.to_csv(new_train_split_file, header=None, index=False, sep=' ')

    new_test_split_file = os.path.join(save_dir, f'test.csv')
    df = pd.DataFrame(test_label_dict.items())
    df.to_csv(new_test_split_file, header=None, index=False, sep=' ')

    ## val == test, simply specify in the code, do not generate the csv file


"""
Total samples in split 1: 9172, train=7333, test=1839
Total samples in split 2: 9172, train=7335, test=1837
Total samples in split 3: 9172, train=7339, test=1833
Total samples in split 4: 9172, train=7340, test=1832
Total samples in split 5: 9172, train=7341, test=1831
"""