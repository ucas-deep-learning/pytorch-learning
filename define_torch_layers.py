  
import torch.nn as nn
from collections import OrderedDict

net1 = nn.Sequential()
net1.add_module('conv1', nn.Conv2d(3,3,3))
net1.add_module('batchnorm', nn.BatchNorm2d(3))
net1.add_module('activation_layer', nn.ReLU())

net2 = nn.Sequential(
    nn.Conv2d(3,3,3),
    nn.BatchNorm2d(3),
    nn.ReLU()
)

net3 = nn.Sequential(
    OrderedDict([
        ('conv1', nn.Conv2d(3, 3, 3)),
        ('bn', nn.BatchNorm2d(3)),
        ('relu', nn.ReLU())
    ])
)

print('第一种方式定义的网络', net1)
print('第二种方式定义的网络', net2)
print('第三种方式定义的网络', net3)
