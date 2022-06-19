import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import cv2  # opencv库
from matplotlib.colors import LinearSegmentedColormap
from numpy.lib.npyio import save
from numpy.lib.type_check import imag
import os
from config import path

'''
处理睡姿数据，可视化，保存图像
'''
# 滤波版本显示比例
clist = ['white', 'white', 'royalblue', 'royalblue', 'lightgreen', 'yellow', 'orangered']
# 调节第二个whilte的比例位置可以抑制噪声的显示
nodes = [0.0, 0.05, 0.05, 0.2, 0.4, 0.5, 1.0]  # 色阶比例

newcmp = LinearSegmentedColormap.from_list('chaos', list(zip(nodes, clist)))
kernel = (55, 55)  # 高斯滤波的核大小

path = path
save_path = os.path.join("./image", "0")  # 新建保存目录
if not os.path.exists(save_path):
    os.makedirs(save_path)

i = 0
for info_txt in os.listdir(path):
    if info_txt == "image":
        continue

    fname = os.path.join(path, info_txt)
    image = np.loadtxt(fname=fname, delimiter=" ", dtype=np.float)
    # 图像阈值过滤，将1000压力值以下置为0
    _, image = cv2.threshold(image, 1000, 1000, cv2.THRESH_TOZERO)

    image = cv2.resize(image, (320, 640), interpolation=cv2.INTER_LINEAR)

    # 高斯去噪
    # 高斯滤波 (7, 7) (15,15) (21,21) (35, 35) (45, 45)  卷积核越大，图像越平滑
    image = cv2.GaussianBlur(image, (45, 45), 0)

    _, image = cv2.threshold(image, 500, 500, cv2.THRESH_TOZERO)

    plt.axis('off')  # 关闭坐标轴
    # 使用色阶显示图像
    plt.grid()
    plt.imshow(image, cmap=newcmp, vmin=0, vmax=4095)  # cmap = 'gray' , interpolation = 'bicubic'
    # plt.colorbar()  # 显示色轴
    i = i + 1
    plt.savefig(os.path.join(save_path, str(i)) + ".jpg", dpi=400)
    print("image output...", i)
