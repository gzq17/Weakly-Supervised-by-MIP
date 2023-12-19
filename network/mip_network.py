import torch
import torch.nn as nn
import torch.nn.functional as F
from .net_module import ResidualBlock, DeResidualBlock, UpConv

class SegmentationModel2Drop(nn.Module):

    def __init__(self, in_channel, out_channel, activate='leakrelu', norm='batch', num_list = [8, 16, 32, 64, 128]):
        super(SegmentationModel2Drop, self).__init__()
        print(activate, norm)
        self.conv1 = ResidualBlock(in_channel, out_channels=num_list[0], stride=1, kernel_size=3, padding=1, activate=activate, norm=norm)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv2 = ResidualBlock(num_list[0], num_list[1], stride=1, kernel_size=3, padding=1, activate=activate, norm=norm)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv3 = ResidualBlock(num_list[1], num_list[2], stride=1, kernel_size=3, padding=1, activate=activate, norm=norm)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv4 = ResidualBlock(num_list[2], num_list[3], stride=1, kernel_size=3, padding=1, activate=activate, norm=norm)
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv5 = ResidualBlock(num_list[3], num_list[4], stride=1, kernel_size=3, padding=1, activate=activate, norm=norm)

        self.upconv1 = UpConv(num_list[4], num_list[3], activate=activate, norm=norm)
        self.deconv1 = DeResidualBlock(num_list[3] * 2, num_list[3], stride=1, kernel_size=3, padding=1, activate=activate, norm=norm)

        self.upconv2 = UpConv(num_list[3], num_list[2], activate=activate, norm=norm)
        self.deconv2 = DeResidualBlock(num_list[2] * 2, num_list[2], stride=1, kernel_size=3, padding=1, activate=activate, norm=norm)

        self.upconv3 = UpConv(num_list[2], num_list[1], activate=activate, norm=norm)
        self.deconv3 = DeResidualBlock(num_list[1] * 2, num_list[1], stride=1, kernel_size=3, padding=1, activate=activate, norm=norm)

        self.upconv4 = UpConv(num_list[1], num_list[0], activate=activate, norm=norm)
        self.deconv4 = DeResidualBlock(num_list[0] * 2, num_list[0], stride=1, kernel_size=3, padding=1, activate=activate, norm=norm)

        self.deconv5 = nn.Conv3d(num_list[0], num_list[0], kernel_size=1, stride=1, bias=True)
        self.pred_prob = nn.Conv3d(num_list[0], out_channel, kernel_size=1, stride=1, bias=True)
        self.pred_soft = nn.Softmax(dim=1)

        self.dropout1 = nn.Dropout(p=0.3)
        self.dropout2 = nn.Dropout(p=0.3)

    def forward(self, x, fea=False):
        conv1 = self.conv1(x)
        pool1 = self.pool1(conv1)

        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)

        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)

        conv4 = self.conv4(pool3)
        pool4 = self.pool4(conv4)

        conv5 = self.conv5(pool4)
        conv5 = self.dropout1(conv5)

        deconv1_1 = self.upconv1(conv5)
        concat_1 = torch.cat((deconv1_1, conv4), dim=1)
        deconv1_2 = self.deconv1(concat_1, deconv1_1)

        deconv2_1 = self.upconv2(deconv1_2)
        concat_2 = torch.cat((deconv2_1, conv3), dim=1)
        deconv2_2 = self.deconv2(concat_2, deconv2_1)

        deconv3_1 = self.upconv3(deconv2_2)
        concat_3 = torch.cat((deconv3_1, conv2), dim=1)
        deconv3_2 = self.deconv3(concat_3, deconv3_1)

        deconv4_1 = self.upconv4(deconv3_2)
        concat_4 = torch.cat((deconv4_1, conv1), dim=1)
        deconv4_2 = self.deconv4(concat_4, deconv4_1)

        deconv5_1 = self.deconv5(deconv4_2)
        deconv5_1 = self.dropout2(deconv5_1)
        pred_soft = self.pred_prob(deconv5_1)
        # pred_soft = self.pred_soft(pred_soft)
        if fea:
            feature = {'0':[pred_soft], '1':[conv1, deconv4_2], '2':[conv2, deconv3_2],
             '3':[conv3, deconv2_2], '4':[conv4, deconv1_2], '5':[conv5]}
            return pred_soft, feature
        else:
            return pred_soft

class SegmentationModel2(nn.Module):

    def __init__(self, in_channel, out_channel, activate='leakrelu', norm='batch', num_list = [8, 16, 32, 64, 128]):
        super(SegmentationModel2, self).__init__()
        print(activate, norm)
        self.conv1 = ResidualBlock(in_channel, out_channels=num_list[0], stride=1, kernel_size=3, padding=1, activate=activate, norm=norm)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv2 = ResidualBlock(num_list[0], num_list[1], stride=1, kernel_size=3, padding=1, activate=activate, norm=norm)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv3 = ResidualBlock(num_list[1], num_list[2], stride=1, kernel_size=3, padding=1, activate=activate, norm=norm)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv4 = ResidualBlock(num_list[2], num_list[3], stride=1, kernel_size=3, padding=1, activate=activate, norm=norm)
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv5 = ResidualBlock(num_list[3], num_list[4], stride=1, kernel_size=3, padding=1, activate=activate, norm=norm)

        self.upconv1 = UpConv(num_list[4], num_list[3], activate=activate, norm=norm)
        self.deconv1 = DeResidualBlock(num_list[3] * 2, num_list[3], stride=1, kernel_size=3, padding=1, activate=activate, norm=norm)

        self.upconv2 = UpConv(num_list[3], num_list[2], activate=activate, norm=norm)
        self.deconv2 = DeResidualBlock(num_list[2] * 2, num_list[2], stride=1, kernel_size=3, padding=1, activate=activate, norm=norm)

        self.upconv3 = UpConv(num_list[2], num_list[1], activate=activate, norm=norm)
        self.deconv3 = DeResidualBlock(num_list[1] * 2, num_list[1], stride=1, kernel_size=3, padding=1, activate=activate, norm=norm)

        self.upconv4 = UpConv(num_list[1], num_list[0], activate=activate, norm=norm)
        self.deconv4 = DeResidualBlock(num_list[0] * 2, num_list[0], stride=1, kernel_size=3, padding=1, activate=activate, norm=norm)

        self.deconv5 = nn.Conv3d(num_list[0], num_list[0], kernel_size=1, stride=1, bias=True)
        self.pred_prob = nn.Conv3d(num_list[0], out_channel, kernel_size=1, stride=1, bias=True)
        self.pred_soft = nn.Softmax(dim=1)

    def forward(self, x, fea=False):
        conv1 = self.conv1(x)
        pool1 = self.pool1(conv1)

        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)

        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)

        conv4 = self.conv4(pool3)
        pool4 = self.pool4(conv4)

        conv5 = self.conv5(pool4)

        deconv1_1 = self.upconv1(conv5)
        concat_1 = torch.cat((deconv1_1, conv4), dim=1)
        deconv1_2 = self.deconv1(concat_1, deconv1_1)

        deconv2_1 = self.upconv2(deconv1_2)
        concat_2 = torch.cat((deconv2_1, conv3), dim=1)
        deconv2_2 = self.deconv2(concat_2, deconv2_1)

        deconv3_1 = self.upconv3(deconv2_2)
        concat_3 = torch.cat((deconv3_1, conv2), dim=1)
        deconv3_2 = self.deconv3(concat_3, deconv3_1)

        deconv4_1 = self.upconv4(deconv3_2)
        concat_4 = torch.cat((deconv4_1, conv1), dim=1)
        deconv4_2 = self.deconv4(concat_4, deconv4_1)

        deconv5_1 = self.deconv5(deconv4_2)
        pred_prob = self.pred_prob(deconv5_1)
        pred_soft = self.pred_soft(pred_prob)
        if fea:
            feature = {'0':[pred_soft], '1':[conv1, deconv4_2], '2':[conv2, deconv3_2],
             '3':[conv3, deconv2_2], '4':[conv4, deconv1_2], '5':[conv5]}
            return pred_soft, feature
        else:
            return pred_soft

class MipSegmentationModel2Drop(nn.Module):

    def __init__(self, in_channel, out_channel, activate='relu', norm='batch', num_list=[8, 16, 32, 64, 128], ndims=2):
        super(MipSegmentationModel2Drop, self).__init__()
        print(activate, norm)
        self.conv1 = ResidualBlock(in_channel, out_channels=num_list[0], stride=1, kernel_size=3, padding=1, activate=activate, norm=norm, ndims=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = ResidualBlock(num_list[0] * 2, num_list[1], stride=1, kernel_size=3, padding=1, activate=activate, norm=norm, ndims=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = ResidualBlock(num_list[1] * 2, num_list[2], stride=1, kernel_size=3, padding=1, activate=activate, norm=norm, ndims=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = ResidualBlock(num_list[2] * 2, num_list[3], stride=1, kernel_size=3, padding=1, activate=activate, norm=norm, ndims=2)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = ResidualBlock(num_list[3] * 2, num_list[4], stride=1, kernel_size=3, padding=1, activate=activate, norm=norm, ndims=2)

        self.upconv1 = UpConv(num_list[4] * 2, num_list[3], activate=activate, norm=norm, ndims=2)
        self.deconv1 = DeResidualBlock(num_list[3] * 2, num_list[3], stride=1, kernel_size=3, padding=1, activate=activate, norm=norm, ndims=2)

        self.upconv2 = UpConv(num_list[3] * 2, num_list[2], activate=activate, norm=norm, ndims=2)
        self.deconv2 = DeResidualBlock(num_list[2] * 2, num_list[2], stride=1, kernel_size=3, padding=1, activate=activate, norm=norm, ndims=2)

        self.upconv3 = UpConv(num_list[2] * 2, num_list[1], activate=activate, norm=norm, ndims=2)
        self.deconv3 = DeResidualBlock(num_list[1] * 2, num_list[1], stride=1, kernel_size=3, padding=1, activate=activate, norm=norm, ndims=2)

        self.upconv4 = UpConv(num_list[1] * 2, num_list[0], activate=activate, norm=norm, ndims=2)
        self.deconv4 = DeResidualBlock(num_list[0] * 2, num_list[0], stride=1, kernel_size=3, padding=1, activate=activate, norm=norm, ndims=2)

        self.deconv5 = nn.Conv2d(num_list[0] * 2, num_list[0], kernel_size=1, stride=1, bias=True)
        self.pred_prob = nn.Conv2d(num_list[0], out_channel, kernel_size=1, stride=1, bias=True)
        self.pred_soft = nn.Softmax(dim=1)

        self.dropout1 = nn.Dropout(p=0.3)
        self.dropout2 = nn.Dropout(p=0.3)

    def forward(self, x, feature):
        conv1 = self.conv1(x)
        conv1_new = torch.cat([conv1, feature['1'][0]], dim=1)
        pool1 = self.pool1(conv1_new)

        conv2 = self.conv2(pool1)
        conv2_new = torch.cat([conv2, feature['2'][0]], dim=1)
        pool2 = self.pool2(conv2_new)

        conv3 = self.conv3(pool2)
        conv3_new = torch.cat([conv3, feature['3'][0]], dim=1)
        pool3 = self.pool3(conv3_new)

        conv4 = self.conv4(pool3)
        conv4_new = torch.cat([conv4, feature['4'][0]], dim=1)
        pool4 = self.pool4(conv4_new)

        conv5 = self.conv5(pool4)
        conv5 = self.dropout1(conv5)
        conv5_new = torch.cat([conv5, feature['5']], dim=1)

        deconv1_1 = self.upconv1(conv5_new)
        concat_1 = torch.cat((deconv1_1, conv4), dim=1)
        deconv1_2 = self.deconv1(concat_1, deconv1_1)
        deconv1_2_new = torch.cat((deconv1_2, feature['4'][1]), dim=1)

        deconv2_1 = self.upconv2(deconv1_2_new)
        concat_2 = torch.cat((deconv2_1, conv3), dim=1)
        deconv2_2 = self.deconv2(concat_2, deconv2_1)
        deconv2_2_new = torch.cat((deconv2_2, feature['3'][1]), dim=1)

        deconv3_1 = self.upconv3(deconv2_2_new)
        concat_3 = torch.cat((deconv3_1, conv2), dim=1)
        deconv3_2 = self.deconv3(concat_3, deconv3_1)
        deconv3_2_new = torch.cat((deconv3_2, feature['2'][1]), dim=1)

        deconv4_1 = self.upconv4(deconv3_2_new)
        concat_4 = torch.cat((deconv4_1, conv1), dim=1)
        deconv4_2 = self.deconv4(concat_4, deconv4_1)
        deconv4_2_new = torch.cat((deconv4_2, feature['1'][1]), dim=1)

        deconv5_1 = self.deconv5(deconv4_2_new)
        deconv5_1 = self.dropout2(deconv5_1)
        pred_prob = self.pred_prob(deconv5_1)
        pred_soft = self.pred_soft(pred_prob)
        return pred_soft

class MipSegmentationModel2(nn.Module):

    def __init__(self, in_channel, out_channel, activate='relu', norm='batch', num_list=[8, 16, 32, 64, 128], ndims=2):
        super(MipSegmentationModel2, self).__init__()
        print(activate, norm)
        self.conv1 = ResidualBlock(in_channel, out_channels=num_list[0], stride=1, kernel_size=3, padding=1, activate=activate, norm=norm, ndims=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = ResidualBlock(num_list[0] * 2, num_list[1], stride=1, kernel_size=3, padding=1, activate=activate, norm=norm, ndims=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = ResidualBlock(num_list[1] * 2, num_list[2], stride=1, kernel_size=3, padding=1, activate=activate, norm=norm, ndims=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = ResidualBlock(num_list[2] * 2, num_list[3], stride=1, kernel_size=3, padding=1, activate=activate, norm=norm, ndims=2)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = ResidualBlock(num_list[3] * 2, num_list[4], stride=1, kernel_size=3, padding=1, activate=activate, norm=norm, ndims=2)

        self.upconv1 = UpConv(num_list[4] * 2, num_list[3], activate=activate, norm=norm, ndims=2)
        self.deconv1 = DeResidualBlock(num_list[3] * 2, num_list[3], stride=1, kernel_size=3, padding=1, activate=activate, norm=norm, ndims=2)

        self.upconv2 = UpConv(num_list[3] * 2, num_list[2], activate=activate, norm=norm, ndims=2)
        self.deconv2 = DeResidualBlock(num_list[2] * 2, num_list[2], stride=1, kernel_size=3, padding=1, activate=activate, norm=norm, ndims=2)

        self.upconv3 = UpConv(num_list[2] * 2, num_list[1], activate=activate, norm=norm, ndims=2)
        self.deconv3 = DeResidualBlock(num_list[1] * 2, num_list[1], stride=1, kernel_size=3, padding=1, activate=activate, norm=norm, ndims=2)

        self.upconv4 = UpConv(num_list[1] * 2, num_list[0], activate=activate, norm=norm, ndims=2)
        self.deconv4 = DeResidualBlock(num_list[0] * 2, num_list[0], stride=1, kernel_size=3, padding=1, activate=activate, norm=norm, ndims=2)

        self.deconv5 = nn.Conv2d(num_list[0] * 2, num_list[0], kernel_size=1, stride=1, bias=True)
        self.pred_prob = nn.Conv2d(num_list[0], out_channel, kernel_size=1, stride=1, bias=True)
        self.pred_soft = nn.Softmax(dim=1)

    def forward(self, x, feature):
        conv1 = self.conv1(x)
        conv1_new = torch.cat([conv1, feature['1'][0]], dim=1)
        pool1 = self.pool1(conv1_new)

        conv2 = self.conv2(pool1)
        conv2_new = torch.cat([conv2, feature['2'][0]], dim=1)
        pool2 = self.pool2(conv2_new)

        conv3 = self.conv3(pool2)
        conv3_new = torch.cat([conv3, feature['3'][0]], dim=1)
        pool3 = self.pool3(conv3_new)

        conv4 = self.conv4(pool3)
        conv4_new = torch.cat([conv4, feature['4'][0]], dim=1)
        pool4 = self.pool4(conv4_new)

        conv5 = self.conv5(pool4)
        conv5_new = torch.cat([conv5, feature['5']], dim=1)

        deconv1_1 = self.upconv1(conv5_new)
        concat_1 = torch.cat((deconv1_1, conv4), dim=1)
        deconv1_2 = self.deconv1(concat_1, deconv1_1)
        deconv1_2_new = torch.cat((deconv1_2, feature['4'][1]), dim=1)

        deconv2_1 = self.upconv2(deconv1_2_new)
        concat_2 = torch.cat((deconv2_1, conv3), dim=1)
        deconv2_2 = self.deconv2(concat_2, deconv2_1)
        deconv2_2_new = torch.cat((deconv2_2, feature['3'][1]), dim=1)

        deconv3_1 = self.upconv3(deconv2_2_new)
        concat_3 = torch.cat((deconv3_1, conv2), dim=1)
        deconv3_2 = self.deconv3(concat_3, deconv3_1)
        deconv3_2_new = torch.cat((deconv3_2, feature['2'][1]), dim=1)

        deconv4_1 = self.upconv4(deconv3_2_new)
        concat_4 = torch.cat((deconv4_1, conv1), dim=1)
        deconv4_2 = self.deconv4(concat_4, deconv4_1)
        deconv4_2_new = torch.cat((deconv4_2, feature['1'][1]), dim=1)

        deconv5_1 = self.deconv5(deconv4_2_new)
        pred_prob = self.pred_prob(deconv5_1)
        pred_soft = self.pred_soft(pred_prob)
        return pred_soft

class MipModelNum1TwoChannel(nn.Module):

    def __init__(self, in_channel, out_channel, activate='relu', norm='batch', num_list = [8, 16, 32, 64, 128], device='cuda', mip_num=3):
        super(MipModelNum1TwoChannel, self).__init__()
        print(activate, norm)
        self.device = device
        self.seg3d = SegmentationModel2(in_channel, 2, activate=activate, norm=norm, num_list=num_list).to(device)
        self.mip_num = mip_num
        self.mip_model0 = MipSegmentationModel2(in_channel, 2, activate=activate, norm=norm, num_list=num_list).to(device)
    
    def forward(self, img3d, mip0, mip1, mip2, mip_index):
        result3d, feature = self.seg3d(img3d, True)
        mip_list = [mip0, mip1, mip2]
        feature_list = []
        for dim in range(0, self.mip_num):
            feature_one = {}
            for jj in range(0, 6):
                now_index = mip_index[dim][jj]
                if jj == 0:
                    feature_one['0'] = feature['0'][0].gather(dim - 3, now_index).squeeze(dim - 3)
                elif jj == 5:
                    feature_one['5'] = feature['5'][0].gather(dim - 3, now_index).squeeze(dim - 3)
                else:
                    one = feature[str(jj)][0].gather(dim - 3, now_index).squeeze(dim - 3)
                    two = feature[str(jj)][1].gather(dim - 3, now_index).squeeze(dim - 3)
                    feature_one[str(jj)] = [one, two]
            feature_list.append(feature_one)
        mip_result_list, feature_list_result = [], []
        for dim in range(0, self.mip_num):
            mip_result = self.mip_model0(mip_list[dim], feature_list[dim])
            mip_result_list.append(mip_result)
            feature_list_result.append(feature_list[dim]['0'])
        return result3d, feature_list_result, mip_result_list

class MipModelNum1TwoChannelDrop(nn.Module):

    def __init__(self, in_channel, out_channel, activate='relu', norm='batch', num_list = [8, 16, 32, 64, 128], device='cuda', mip_num=3):
        super(MipModelNum1TwoChannelDrop, self).__init__()
        print(activate, norm)
        self.device = device
        self.seg3d = SegmentationModel2Drop(in_channel, 2, activate='relu', norm='instance', num_list=num_list).to(device)
        self.mip_num = mip_num
        self.mip_model0 = MipSegmentationModel2Drop(in_channel, 2, activate='relu', norm='instance', num_list=num_list).to(device)
    
    def forward(self, img3d, mip0, mip1, mip2, mip_index, infer=False):
        result3d, feature = self.seg3d(img3d, True)
        if infer:
            return result3d
        mip_list = [mip0, mip1, mip2]
        feature_list = []
        for dim in range(0, self.mip_num):
            feature_one = {}
            for jj in range(0, 6):
                if jj == 0:
                    feature_one['0'] = feature['0'][0].gather(dim - 3, mip_index[dim][0].to(self.device)).squeeze(dim - 3)
                elif jj == 5:
                    feature_one['5'] = feature['5'][0].gather(dim - 3, mip_index[dim][5].to(self.device)).squeeze(dim - 3)
                else:
                    one = feature[str(jj)][0].gather(dim - 3, mip_index[dim][jj].to(self.device)).squeeze(dim - 3)
                    two = feature[str(jj)][1].gather(dim - 3, mip_index[dim][jj].to(self.device)).squeeze(dim - 3)
                    feature_one[str(jj)] = [one, two]
            feature_list.append(feature_one)
        mip_result_list, feature_list_result = [], []
        for dim in range(0, self.mip_num):
            mip_result = self.mip_model0(mip_list[dim], feature_list[dim])
            mip_result_list.append(mip_result)
            feature_list_result.append(feature_list[dim]['0'])
        return result3d, feature_list_result, mip_result_list
