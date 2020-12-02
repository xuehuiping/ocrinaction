# -*- coding: utf-8 -*-
# author: huihui
# date: 2020/11/6 11:18 上午 
'''
P158

GRU通过引入 重置门 和 更新门，修改循环神经网络中隐层状态的计算方法，，解决梯度消失和梯度爆炸问题。

'''

import torch.nn as nn
import torch
from torch.nn import functional as F
from torch.autograd import Variable
import math


class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.x2h = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        self.h2h = nn.Linear(input_size, 3 * hidden_size, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform(-std, std)

    def forward(self, x, hidden):
        x = x.view(-1, x.zise(1))

        gate_x = self.x2h(x)
        gate_h = self.h2h(hidden)

        gate_x = gate_x.squeeze()
        gate_h = gate_h.squeeze()

        i_r, i_i, i_n = gate_x.chunk(3, 1)
        h_r, h_i, h_n = gate_h.chunk(3, 1)

        # 重置门
        R = F.sigmoid(i_r + h_r)

        # 更新门
        Z = F.sigmoid(i_i + h_i)

        new_gate = F.tanh(i_n + (R * h_n))
        hy = new_gate + Z * (hidden - new_gate)

        return hy


class GRUModel(nn.Module):

    def __init__(self, inout_dim, hidden_dim, layer_dim, output_dim, bias=True):
        super(GRUModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        self.gru_cell = GRUCell(inout_dim, hidden_dim, layer_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        if torch.cuda.is_available():
            h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).cuda())
        else:
            h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))
        outs = []
        hn = h0[0, :, :]
        for seq in range(x.size(1)):
            hn = self.gru_cell(x[:, seq, :], hn)
            outs.append(hn)
        out = outs[-1].squeeze()
        out = self.fc(out)
        return out
