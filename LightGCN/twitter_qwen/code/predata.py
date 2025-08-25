import pandas as pd
import numpy as np
import random

if world.config['view'] == "r":
    filename = "../data/user_f/2_repost/all_repost.txt"
    train_file = "../data/user_f/2_repost/train.txt"
    test_file = "../data/user_f/2_repost/test.txt"
else:
    filename = "../data/user_f/3_comment/all_comment.txt"
    train_file = "../data/user_f/3_comment/train.txt"
    test_file = "../data/user_f/3_comment/test.txt"

data = pd.read_csv(filename,header=None).to_numpy()
length = len(data)
index = [i for i in range(length)]
np.random.shuffle(index)

train_index = index[:int(length*0.8)]
test_index = index[int(length*0.8):]

train_data = data[train_index]
test_data = data[test_index]


with open(train_file, 'w') as file:
    for row in train_data:
        if world.config['view'] == "r" or world.config['view'] == "c":
            user, att_user, rela = row[0].strip().split('\t')
            print(row[0])
            file.write(f"{user} {att_user} {rela}\n")
        else:
            user, att_user = row[0].strip().split('\t')
            print(row[0])
            file.write(f"{user} {att_user}\n")
    print('write train file over!')

# 写入测试集文件
with open(test_file, 'w') as file:
    for row in test_data:
        if world.config['view'] == "r" or world.config['view'] == "c":
            user, att_user, rela = row[0].strip().split('\t')
            file.write(f"{user} {att_user} {rela}\n")
        else:
            user, att_user = row[0].strip().split('\t')
            file.write(f"{user} {att_user}\n")
    print('write test file over!')
