

import torch
import torch.nn as nn
from torch.nn import functional as F


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=3, stride=stride,
                     padding=1, bias=False)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1,
                 downsample=None):
        super(ResidualBlock, self).__init__()

        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample  # 是否进行下采样，保证残差相加的时候通道维度一致

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    """
    ResNet(ResidualBlock, [2,2,2])
    """

    def __init__(self, block, layers, num_classes=7):
        super(ResNet, self).__init__()
        self.in_channels = 16  # 输入block的通道数量

        self.conv = conv3x3(1, 16)  # 第一个卷积层，提升通道
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(16, 16, 2)
        self.bn2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU(inplace=True)

        self.layer1 = self.make_layer(block, 16, layers[0], 2)
        self.layer2 = self.make_layer(block, 32, layers[1], 2)
        self.layer3 = self.make_layer(block, 64, layers[2], 2)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes)

    def make_layer(self, block, out_channels, blocks, stride=1):
        """
        blocks:表示该层使用基本基本的blcok进行堆叠
        """
        downsample = None

        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):

        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)

        #         out = self.conv2(out)
        #         out = self.bn2(out)
        #         out = self.relu2(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)  # [-1, 64, 1, 1]

        out = out.view(out.size(0), -1)  # 将每个图像的特征图转成一维的特征向量
        out = self.fc(out)
        return out


def num_flat_features(x):
    # x.size()返回值为(256, 16, 5, 5)，size的值为(16, 5, 5)，256是batch_size
    size = x.size()[1:]  # x.size返回的是一个元组，size表示截取元组中第二个开始的数字
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


class LeNet(nn.Module):
    """
    LeNet
    """

    def __init__(self):
        super(LeNet, self).__init__()
        # self.in_channels = 16  # 输入block的通道数量

        # 定义卷积层，1个输入通道，6个输出通道，5*5的卷积filter，外层补上了两圈0,因为输入的是32*32
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv11 = nn.Conv2d(6, 6, 5)
        # 第二个卷积层，6个输入，16个输出，5*5的卷积filter
        self.conv2 = nn.Conv2d(6, 16, 5)

        # 最后是三个全连接层
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 7)

    def forward(self, x):
        '''前向传播函数'''
        # 先卷积，然后调用relu激活函数，再最大值池化操作
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv11(x)), (2, 2))
        # 第二次卷积+池化操作
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        # 重新塑形,将多维数据重新塑造为二维数据，256*400
        x = x.view(-1, num_flat_features(x))
        # print('size', x.size())
        # 第一个全连接
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class VGG(nn.Module):
    def __init__(self, num_classes=7):
        super(VGG, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(16, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(1024 * 2, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class AlexNet(nn.Module):
    def __init__(self, num_classes=7, init_weights=False):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(  # 打包
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # output[16, 64, 64]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # output[16, 32, 32]
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # output[32, 32, 32]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # output[32, 16, 16]
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # output[64, 8, 8]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # output[64, 4, 4]
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            # 全链接
            nn.Linear(49 * 8 * 8, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)  # 展平   或者view()
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  # 何教授方法
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)  # 正态分布赋值
                nn.init.constant_(m.bias, 0)

def vgg10():
    model = VGG()
    return model


def resnet():
    model = ResNet(ResidualBlock, [2, 2, 2])
    return model


def alexnet():
    model = AlexNet()
    return model

def lenet():
    model = LeNet()
    return model
