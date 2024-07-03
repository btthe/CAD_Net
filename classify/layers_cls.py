import numpy as np
import torch
from torch import nn



class PostRes(nn.Module):
    def __init__(self, n_in, n_out, stride=1):
        super(PostRes, self).__init__()
        self.conv1 = nn.Conv3d(n_in, n_out, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm3d(n_out)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(n_out, n_out, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(n_out)

        # self.globalAvgPool = nn.AdaptiveAvgPool3d((1, 1, 1))
        # self.fc1 = nn.Linear(in_features=n_out, out_features=round(n_out / 16))
        # self.fc2 = nn.Linear(in_features=round(n_out / 16), out_features=n_out)
        # self.sigmoid = nn.Sigmoid()
        # self.convfc1 = nn.Conv3d(n_out, round(n_out / 16), kernel_size=1, stride=1, padding=0)
        # self.convfc2 = nn.Conv3d(round(n_out / 16), n_out, kernel_size=1, stride=1, padding=0)

        if stride != 1 or n_out != n_in:
            self.shortcut = nn.Sequential(
                nn.Conv3d(n_in, n_out, kernel_size=1, stride=stride),
                nn.BatchNorm3d(n_out))
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        # original_out = out
        # out = self.globalAvgPool(out)
        # out = self.convfc1(out)
        # out = self.relu(out)
        # out = self.convfc2(out)
        # out = self.sigmoid(out)
        # out = out.view(out.size(0), out.size(1), 1, 1, 1)
        # out = out * original_out

        out += residual
        out = self.relu(out)
        return out

class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(BinaryFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average
        # self.sigmoid = nn.Sigmoid()  # 激活函数
        print ("FOCAL LOSS", gamma, alpha)

    def forward(self, input, target, train=True):

        if input.dim() == 1:
            input = input.unsqueeze(1)
        if target.dim() == 1:
            target = target.unsqueeze(1)

        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C

        #target = target.view(-1,1)
        # input = self.sigmoid(input)
        # target = target.float()
        pt = input * target + (1 - input) * (1 - target)
        logpt = pt.log()
        at = self.alpha * target + (1 - self.alpha) * (1 - target)
        logpt = logpt * at

        loss = -1 * (1 - pt) ** self.gamma * logpt

        return loss.mean()







