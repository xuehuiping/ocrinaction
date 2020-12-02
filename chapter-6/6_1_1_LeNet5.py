# -*- coding: utf-8 -*-
# author: huihui
# date: 2020/11/6 10:31 上午 
'''1988年提出，MINIST'''

import torch.nn as nn
from torch.nn import functional as F


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()

        # 卷积层1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)

        # 卷积层2
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)

        # 全连接层1
        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)

        # 全连接层2
        self.fc2 = nn.Linear(in_features=120, out_features=84)

        # 全连接层3
        self.fc3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
