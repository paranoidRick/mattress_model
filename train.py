import datetime
import pickle
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset

import model
from cmPlot import confusion_matrix, plot_confusion_matrix
from config import *
from dataSet import SleepPosDataset
from earlyStopping import EarlyStopping

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('use device: ', device)

# 构造dataloader
train_dataset = SleepPosDataset(os.path.join(path, "train.txt"))
test_dataset = SleepPosDataset(os.path.join(path, "test.txt"))

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

# 导入模型
model = model.resnet()
# 交叉熵损失函数
criterion = nn.CrossEntropyLoss()
# 亚当优化器
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


total_step = len(train_loader)
print("total_step: ", total_step)

curr_lr = lr  # 当前学习率
epoch_loss = []
test_acces = []
log_step = 50  # 每20步报告训练情况
early_stopping = EarlyStopping(patience=5, verbose=True, checkpoint_name="sleep_res_pos.pt")

start_time = datetime.datetime.now()
for epoch in range(1, num_epochs):
    curr_loss = []
    model.train()
    # 训练部分
    for i, (images, labels) in enumerate(train_loader, 1):
        # 前向传播
        outputs = model(images)
        # 计算损失
        loss = criterion(outputs, labels)
        curr_loss.append(loss.item())

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 计算该epoch的平均损失
    epoch_loss.append(np.mean(curr_loss))
    train_loss = np.average(epoch_loss)
    early_stopping(train_loss, model)

    # 学习率衰减
    if (epoch) % 10 == 0:
        curr_lr /= 3
        update_lr(optimizer, curr_lr)

    # 在测试集上检验效果
    test_loss = 0
    test_acc = 0
    model.eval()  # 将模型改为预测模式
    # 创建一个空矩阵存储混淆矩阵
    # conf_matrix = torch.zeros(7, 7)
    # 每次迭代都是处理一个小批量的数据
    for images, labels in test_loader:
        images = Variable(images)  # torch中训练需要将其封装即Variable
        labels = Variable(labels)  # 此处为标签

        out = model(images)  # 经网络输出的结果
        loss = criterion(out, labels)  # 得到误差
        # 绘制混淆矩阵
        # prediction = torch.max(out, 1)[1]  # 得到预测结果
        # conf_matrix = confusion_matrix(prediction, labels=labels, conf_matrix=conf_matrix)
        # # conf_matrix需要是numpy格式
        # plot_confusion_matrix(conf_matrix.numpy(), classes=sleep_type, normalize=True,
        #                       title='Normalized confusion matrix')
        # 记录误差
        test_loss += loss.item()

        # 记录准确率
        out_t = out.argmax(dim=1)  # 取出预测的最大值的索引
        num_correct = (out_t == labels).sum().item()  # 判断是否预测正确
        acc = num_correct / images.shape[0]  # 计算准确率
        test_acc += acc

    test_acces.append(test_acc / len(test_loader))
    print('Eval Loss: {:.6f}, Eval Acc: {:.6f}'.format(test_loss / len(test_loader), test_acc / len(test_loader)))

end_time = datetime.datetime.now()
print(end_time - start_time)
# 持久化test_acc
# resnet5_acc_file = open("resnet5_acc", "wb")
# pickle.dump(test_acces, resnet5_acc_file)

plt.plot(np.arange(len(test_acces)), test_acces, label="test acc")
# 显示图例
plt.legend()
plt.xlabel('epochs')
plt.title('Model accuracy')
plt.show()
