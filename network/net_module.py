import torch
import torch.nn as nn
import torch.nn.functional as F

class BnReluConv(nn.Module):

    def __init__(self, in_channels, out_channels, stride, kernel_size, padding, activate='relu', norm='batch', ndims=3):
        super(BnReluConv, self).__init__()
        if norm == 'batch':
            Norm_ = getattr(nn, 'BatchNorm%dd' % ndims)
            self.norm = Norm_(in_channels)
        else:
            Norm_ = getattr(nn, 'InstanceNorm%dd' % ndims)
            self.norm = Norm_(in_channels)
        if activate == 'relu':
            self.relu = nn.ReLU(inplace=False)
        elif activate == 'leakrelu':
            self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        elif activate == 'prelu':
            self.relu = nn.PReLU()
        else:
            self.relu = None
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.conv = Conv(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        x = self.norm(x)
        x = self.relu(x)
        x = self.conv(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, kernel_size, padding, activate='relu', norm='batch', ndims=3):
        super(ResidualBlock, self).__init__()
        self.bn_relu_conv1 = BnReluConv(in_channels, out_channels, stride, kernel_size, padding, activate, norm, ndims)
        self.bn_relu_conv2 = BnReluConv(out_channels, out_channels, stride, kernel_size, padding, activate, norm, ndims)

    def forward(self, x):
        y = self.bn_relu_conv1(x)
        residual = y
        z = self.bn_relu_conv2(y)
        return z + residual

class DeResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride, kernel_size, padding, activate='relu', norm='batch', ndims=3):
        super(DeResidualBlock, self).__init__()
        self.bn_relu_conv1 = BnReluConv(in_channels, out_channels, stride, kernel_size, padding, activate, norm, ndims)
        self.bn_relu_conv2 = BnReluConv(out_channels, out_channels, stride, kernel_size, padding, activate, norm, ndims)

    def forward(self, x1, x2):
        y = self.bn_relu_conv1(x1)
        y = self.bn_relu_conv2(y)
        return y + x2

class UpConv(nn.Module):

    def __init__(self, in_channels, out_channels, activate='relu', norm='batch', ndims=3):
        super(UpConv, self).__init__()
        Conv = getattr(nn, 'ConvTranspose%dd' % ndims)
        self.conv = Conv(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        if norm == 'batch':
            Norm_ = getattr(nn, 'BatchNorm%dd' % ndims)
            self.norm = Norm_(out_channels)
        else:
            Norm_ = getattr(nn, 'InstanceNorm%dd' % ndims)
            self.norm = Norm_(out_channels)
        if activate == 'relu':
            self.relu = nn.ReLU(inplace=False)
        elif activate == 'leakrelu':
            self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        elif activate == 'prelu':
            self.relu = nn.PReLU()
        else:
            self.relu = None

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x
