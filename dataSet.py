import os
import random

import numpy as np

from torch.utils.data import Dataset
from torchvision import transforms

from config import path, sleep_type

dataset_txt = open(os.path.join(path, "dataset.txt"), 'w')  # 如果文件不存在自动创建
train_index_txt = open(os.path.join(path, "train.txt"), 'w')
test_index_txt = open(os.path.join(path, "test.txt"), 'w')

ration = 0.2  # 训练集和测试集比例

# 从文件夹中读出文件路径 -> .\Datas\0\Info_10792.txt 0
for class_ in sleep_type:
    data_dir = os.path.join(path, class_)
    img = os.listdir(data_dir)
    img_len = len(img)
    test_list = []
    for _ in range(0, int(img_len * ration)):
        test_list.append(random.randint(0, img_len))

    for i, data in enumerate(img):
        img = np.loadtxt(fname=os.path.join(data_dir, data), delimiter=" ", dtype=np.float64)
        line = os.path.join(data_dir, data) + ' ' + class_ + '\n'
        dataset_txt.write(line)

        if i in test_list:
            test_index_txt.write(line)
        else:
            train_index_txt.write(line)

# 关闭资源
train_index_txt.close()
test_index_txt.close()
dataset_txt.close()

class SleepPosDataset(Dataset):
    def __init__(self, txt_path, transform=None, target_transform=None):

        fh = open(txt_path, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], int(words[1])))

        self.imgs = imgs  # 最主要就是要生成这个list， 然后DataLoader中给index，通过getitem读取图片数据
        self.transform = transforms.ToTensor()
        self.target_transform = target_transform

    def __getitem__(self, index):
        fname, label = self.imgs[index]
        img = np.loadtxt(fname=fname, delimiter=" ", dtype=np.float64)

        # 简单滤波
        for row in range(img.shape[0]):
            for col in range(img.shape[1]):
                if img[row][col] < 500:
                    img[row][col] = 0

        img = np.hstack((img, np.zeros_like(img)))
        img = img[:, :, np.newaxis]

        if self.transform is not None:
            img = self.transform(img).float()  # 在这里做transform，转为tensor等等

        return img, label

    def __len__(self):
        return len(self.imgs)
