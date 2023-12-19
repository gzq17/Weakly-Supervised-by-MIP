from curses import echo
import os
import random
import argparse
import time
import numpy as np
import torch, sys, copy
import torch.nn.functional as F
torch.cuda.manual_seed(1234)
from utils.dataset_mip import MipDataset2TwoChannelFold
from torch.utils.data import DataLoader
from utils.loss import WeightedCorssLoss3dTwoChannel, Dice2DTwoChannel
from network.mip_network import MipModelNum1TwoChannel, MipModelNum1TwoChannelDrop
import SimpleITK as sitk
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
root_path = '/disk1/guozhanqiang/Cerebral/mip_data/'

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', type=str, default='model/work_expansion/mra_data1_model/')
parser.add_argument('--old_model', type=str, default='')
parser.add_argument('--rec_path', type=str, default='reconstruction_label2_one_new/')
parser.add_argument('--num_mip', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=1, help='batch size (default: 1)')
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--seg_lr', type=float, default=1e-3)
parser.add_argument('--num_fea_list', type=float, default=[2, 6, 16, 32, 64, 128])
parser.add_argument('--drop', type=bool, default=False)
parser.add_argument('--all_fold', type=int, default=0)
parser.add_argument('--one_fold', type=int, default=-1)
parser.add_argument('--fine-tune', type=bool, default=False)
parser.add_argument('--val_label', type=bool, default=False)
args = parser.parse_args()
device = 'cuda'
mip_index_dir = {}
if args.fine_tune:
    args.old_model = 'model/work_expansion/mra_data1_model/two_channel/one_model1/best.pth'

def adjust_lr(optimizer, epoch, lr):
    lr_c = lr * ((1 - epoch/(args.epochs + 1)) ** 0.9)
    for p in optimizer.param_groups:
        p['lr'] = lr_c

def get_segmentation_model(inchannel, load_name=None):
    if args.drop:
        print('drop')
        seg_model = MipModelNum1TwoChannelDrop(inchannel, 2, activate='relu', norm='instance', num_list=args.num_fea_list[1:], mip_num=args.num_mip)
    else:
        seg_model = MipModelNum1TwoChannel(inchannel, 2, activate='relu', norm='instance', num_list=args.num_fea_list[1:], mip_num=args.num_mip)
    if load_name is not None:
        print('load')
        seg_model.load_state_dict(torch.load(load_name))
    return seg_model.to(device)

def train(train_loader, val_loader=None):
    loss_file_name = model_dir + 'loss.txt'
    loss_file = open(loss_file_name, 'w')
    for arg in vars(args):
        loss_file.write(f'{arg}: {getattr(args, arg)}\n')
    if args.old_model != '':
        seg_model = get_segmentation_model(inchannel=1, load_name = root_path + args.old_model)
    else:
        seg_model = get_segmentation_model(inchannel=1)
    seg_model.train()
    # seg_optimizer = torch.optim.SGD(seg_model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)
    seg_optimizer = torch.optim.Adam(seg_model.parameters(), lr=args.seg_lr)
    val_info, val_dice = val(seg_model, val_loader, 0)
    seg_model.train()
    val_dice_max = val_dice
    print(val_dice_max, val_dice)
    epoch_max_index = 0
    loss_name = ['loss_fore', 'loss_ctr', 'loss_back', 'result3d_mip_dice_loss', 'result_mip_dice_loss', 'loss_sum']
    for epoch in range(0, args.epochs):
        loss_list_sum = [0, 0, 0, 0, 0, 0]
        if epoch % 10 == 0:
            adjust_lr(seg_optimizer, epoch, args.seg_lr)
        for i, data in enumerate(train_loader):
            if epoch == 0:
                for name in data['name']:
                    loss_file.write(name + '   ')
            img3d = data['img3d'].to(device).float()
            lbl3d = data['lbl3d'].to(device).float()
            lbl3d_fore = data['lbl3d_fore'].to(device).float()
            lbl3d_ctr = data['lbl3d_ctr'].to(device).float()
            lbl3d_back = data['lbl3d_back'].to(device).float()
            mip_img0, mip_lbl0 = data['mip_img0'].to(device).float(), data['mip_lbl0'].to(device).float()
            mip_label = [mip_lbl0]
            mip_index0 = mip_index_dir[data['name'][0]]
            mip_index = [mip_index0]

            result3d, result3d_mip, result_mip = seg_model(img3d, mip_img0, None, None, mip_index)
            loss_fore, loss_ctr, loss_back = WeightedCorssLoss3dTwoChannel(lbl3d_fore, lbl3d_ctr, lbl3d_back, result3d)
            result3d_mip_dice_loss = Dice2DTwoChannel(mip_label, result3d_mip, num=args.num_mip)
            result_mip_dice_loss = Dice2DTwoChannel(mip_label, result_mip, num=args.num_mip)
            loss_sum = loss_fore + loss_ctr + loss_back
            loss_sum = loss_sum + result3d_mip_dice_loss + result_mip_dice_loss
            seg_optimizer.zero_grad()
            loss_sum.backward()
            seg_optimizer.step()
            loss_list_one = [float(loss_fore), float(loss_ctr), float(loss_back),
                            float(result3d_mip_dice_loss), float(result_mip_dice_loss), float(loss_sum)]
            for ii in range(0, len(loss_name)):
                loss_list_sum[ii] = loss_list_sum[ii] + loss_list_one[ii]
            if i % 5 == 0:
                info = 'epoch:{}/{}, {}, '.format(epoch, args.epochs, i)
                for ii in range(0, len(loss_name)):
                    info += loss_name[ii] + ':{:.4f}  '.format(loss_list_one[ii])
                print(info)
                loss_file.write(info + '\n')
                loss_file.flush()
        print(i + 1)
        val_info, val_dice = val(seg_model, val_loader, epoch)
        seg_model.train()
        info_epoch = 'epoch:{}/{} '.format(epoch, args.epochs)
        for ii in range(0, len(loss_name)):
            info_epoch += loss_name[ii] + ':{:.4f}  '.format(loss_list_sum[ii] / (i + 1))
        print(info_epoch + '\n')
        if val_dice > val_dice_max:
            val_dice_max = val_dice
            epoch_max_index = epoch
            torch.save(seg_model.state_dict(), model_dir + 'best.pth')
        loss_file.write(info_epoch + '\n')
        loss_file.write(val_info + '\n')
        loss_file.write('epoch_max_index:{}, val_dice_max:{:.2f}\n'.format(epoch_max_index, val_dice_max))
        loss_file.flush()
        print(epoch_max_index, val_dice_max)
        if (epoch + 1) % 200 == 0:
            torch.save(seg_model.state_dict(), model_dir + str(epoch + 1) +'.pth')
        else:
            torch.save(seg_model.state_dict(), model_dir + 'last.pth')
           
def val(model, val_loader, epoch):
    model.eval()
    dice_sum = 0
    dice_mip_sum = 0
    loss_list_sum = [0, 0, 0, 0, 0, 0]
    loss_name = ['loss_fore', 'loss_ctr', 'loss_back', 'result3d_mip_dice_loss', 'result_mip_dice_loss', 'loss_sum']
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            img3d = data['img3d'].to(device).float()
            lbl3d = data['lbl3d'].to(device).float()
            lbl3d_fore = data['lbl3d_fore'].to(device).float()
            lbl3d_ctr = data['lbl3d_ctr'].to(device).float()
            lbl3d_back = data['lbl3d_back'].to(device).float()
            mip_img0, mip_lbl0 = data['mip_img0'].to(device).float(), data['mip_lbl0'].to(device).float()
            mip_label = [mip_lbl0]
            mip_index0 = mip_index_dir[data['name'][0]]
            mip_index = [mip_index0]

            result3d, result3d_mip, result_mip = model(img3d, mip_img0, None, None, mip_index)
            loss_fore, loss_ctr, loss_back = WeightedCorssLoss3dTwoChannel(lbl3d_fore, lbl3d_ctr, lbl3d_back, result3d)
            result3d_mip_dice_loss = Dice2DTwoChannel(mip_label, result3d_mip, num=args.num_mip)
            result_mip_dice_loss = Dice2DTwoChannel(mip_label, result_mip, num=args.num_mip)
            result_mip_0, mip_label_0 = result_mip[0], mip_label[0]
            result_mip_0 = torch.argmax(result_mip_0, dim=1).unsqueeze(1)
            result3d = torch.argmax(result3d, dim=1).unsqueeze(1)
            dice_one_mip = 2.0 * (((mip_label_0 == 1) & (result_mip_0 == 1)).sum()) / ((mip_label_0 == 1).sum() + (result_mip_0 == 1).sum())
            dice_mip_sum += dice_one_mip
            if args.val_label:
                dice_one = 2.0 * (((lbl3d == 1) & (result3d == 1)).sum()) / ((lbl3d == 1).sum() + (result3d == 1).sum())
            else:
                dice_one = 2.0 * (((lbl3d_fore == 1) & (result3d == 1)).sum()) / ((lbl3d_fore == 1).sum() + (result3d == 1).sum())
            dice_sum += dice_one
            loss_sum = loss_fore + loss_ctr + loss_back
            loss_sum = loss_sum + result3d_mip_dice_loss + result_mip_dice_loss
            loss_list_one = [float(loss_fore), float(loss_ctr), float(loss_back), float(result3d_mip_dice_loss), float(result_mip_dice_loss), float(loss_sum)]
            for ii in range(0, len(loss_name)):
                loss_list_sum[ii] = loss_list_sum[ii] + loss_list_one[ii]
    val_info = 'val info, epoch:{} '.format(epoch)
    for ii in range(0, len(loss_name)):
        val_info += loss_name[ii] + ':{:.4f}  '.format(loss_list_sum[ii] / (i + 1))
    val_info += 'val mip dice:{:.2f}'.format(float(dice_mip_sum / (i + 1) * 100))
    val_info += 'val dice:{:.2f}'.format(float(dice_sum / (i + 1) * 100))
    print(val_info + '\n')
    return val_info, float(dice_sum / (i + 1) * 100)

def main():
    print(args.num_mip)
    if args.all_fold == 0:
        fold_index = '' + '.txt'
    else:
        fold_index = '_fold' + str(args.all_fold) + '.txt'
    if args.drop:
        fold_one = args.one_fold
    else:
        fold_one = None
    train_set = MipDataset2TwoChannelFold('data_txt/train' + fold_index, root_path + 'mra_data1/', feature_num=args.num_fea_list,
        rec_path=args.rec_path, suffix='-', fold=fold_one)
    for i in range(0, train_set.__len__()):
        data = train_set.__getitem__(i)
        mip_index0 = data['mip_index0']
        for j in range(0, len(mip_index0)):
            mip_index0[j] = mip_index0[j].to('cuda').unsqueeze(0)
        mip_index_dir[data['name']] = mip_index0
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
    val_set = MipDataset2TwoChannelFold('data_txt/val' + fold_index, root_path + 'mra_data1/', feature_num=args.num_fea_list,
        rec_path='reconstruction_label2_one_new/', suffix='-', add=1)
    for i in range(0, val_set.__len__()):
        data = val_set.__getitem__(i)
        mip_index0 = data['mip_index0']
        for j in range(0, len(mip_index0)):
            mip_index0[j] = mip_index0[j].to('cuda').unsqueeze(0)
        mip_index_dir[data['name']] = mip_index0
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)
    print('train num:{}'.format(train_set.__len__()))
    train(train_loader, val_loader)
        
if __name__ == '__main__':
    args.num_mip = 1
    if args.all_fold == 0:
        fold_index = '/'
    else:
        fold_index = '_fold' + str(args.all_fold) + '/'
    model_dir_p = root_path + args.model_dir + 'two_channel' + fold_index
    os.makedirs(model_dir_p, exist_ok=True)
    if args.drop:
        model_dir = model_dir_p + 'one_model1_drop_fold' +  str(args.one_fold) + '/'
    else:
        if args.fine_tune:
            model_dir = model_dir_p + 'one_model1-init/'
        else:
            model_dir = model_dir_p + 'one_model1/'
    os.makedirs(model_dir, exist_ok=True)
    main()
