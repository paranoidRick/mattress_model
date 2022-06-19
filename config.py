# 配置文件 —— 超参设置
from torch import nn

# 床垫图片文件路径
path = "D:\\pyCode\\睡姿采集数据\\5Pos"
# 睡姿类型: 0坐立 1右侧树干型 2右侧胎儿型 3仰卧 4左侧树干型 5左侧胎儿型 6俯卧
# sleep_type = ['0', '1', '2', '3', '4', '5', '6']
# 睡姿类型: 平躺 右侧卧 左侧卧 坐立 俯卧
sleep_type = ['0', '1', '2', '3', '4']
# 批数量大小
batch_size = 64
# 学习率
lr = 0.001
# 训练轮数
num_epochs = 25
