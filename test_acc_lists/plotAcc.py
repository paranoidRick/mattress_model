import pickle

import numpy as np
import matplotlib.pyplot as plt

resnet_acc = open('resnet_acc', 'rb')
lenet_acc = open('lenet_acc', 'rb')
alex_acc = open('alex_acc', 'rb')
vgg_acc = open('vgg_acc', 'rb')

resnet_list = pickle.load(resnet_acc)
lenet_list = pickle.load(lenet_acc)
alex_list = pickle.load(alex_acc)
print(lenet_list)
vgg_list = pickle.load(vgg_acc)

plt.plot(np.arange(len(resnet_list)), resnet_list, label="matNet_acc")
plt.plot(np.arange(len(lenet_list)), lenet_list, label="2-layerCNN_acc" , linestyle='dashed')
plt.plot(np.arange(len(alex_list)), alex_list, label="AlexNet_acc", linestyle='dashdot')
plt.plot(np.arange(len(vgg_list)), vgg_list, label="vgg_acc", linestyle='dotted')
# 显示图例
plt.legend()
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
plt.xticks(np.arange(0, 25, 5))
plt.xlabel('Training Epoch')
plt.ylabel("Accuracy")
plt.show()
