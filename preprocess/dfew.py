# *_*coding:utf-8 *_*
"""
cmd:
cd preprocess
python voxceleb2.py
"""

import os
import pandas as pd

# change 'data_path' to yours
dataset = "DFEW"
data_path = os.path.expanduser(f'~/AC/Dataset/{dataset}')
split_dir = os.path.join(data_path, 'EmoLabel_DataSplit')
video_dir = os.path.join(data_path, 'Clip/jpg_256') # 'jpg_256' <=> 'clip_224x224' in the official dataset

num_splits = 5
for split in range(1, 1 + num_splits):
    save_dir = f'../saved/data/dfew/org/split0{split}' # org: original frames in jpg_256, not temporally aligned in jpg_224_16f
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # read
    train_split_file = os.path.join(split_dir, f'train(single-labeled)/set_{split}.csv')
    df = pd.read_csv(train_split_file)
    train_label_dict = dict(zip(df.video_name, df.label))

    test_split_file = os.path.join(split_dir, f'test(single-labeled)/set_{split}.csv')
    df = pd.read_csv(test_split_file)
    test_label_dict = dict(zip(df.video_name, df.label))

    train_label_dict = {os.path.join(video_dir, f'{v:05d}'):(l-1) for v,l in train_label_dict.items()}
    test_label_dict = {os.path.join(video_dir, f'{v:05d}'):(l-1) for v,l in test_label_dict.items()}

    total_samples = len(train_label_dict) + len(test_label_dict)
    print(f'Total samples in split {split}: {total_samples}, train={len(train_label_dict)}, test={len(test_label_dict)}')

    # write
    new_train_split_file = os.path.join(save_dir, f'train.csv')
    df = pd.DataFrame(train_label_dict.items())
    df.to_csv(new_train_split_file, header=None, index=False, sep=' ')

    new_test_split_file = os.path.join(save_dir, f'test.csv')
    df = pd.DataFrame(test_label_dict.items())
    df.to_csv(new_test_split_file, header=None, index=False, sep=' ')

    ## val == test
    new_val_split_file = os.path.join(save_dir, f'val.csv')
    df = pd.DataFrame(test_label_dict.items())
    df.to_csv(new_val_split_file, header=None, index=False, sep=' ')


"""
Total samples in split 1: 11697, train=9356, test=2341
Total samples in split 2: 11697, train=9356, test=2341
Total samples in split 3: 11697, train=9357, test=2340
Total samples in split 4: 11697, train=9358, test=2339
Total samples in split 5: 11697, train=9361, test=2336
"""