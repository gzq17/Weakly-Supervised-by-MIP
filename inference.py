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
parser.add_argument('--num_mip', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=1, help='batch size (default: 1)')
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--seg_lr', type=float, default=1e-3)
parser.add_argument('--drop', type=bool, default=False)
parser.add_argument('--num_fea_list', type=float, default=[2, 6, 16, 32, 64, 128])
parser.add_argument('--all_fold', type=int, default=0)
args = parser.parse_args()
device = 'cuda'
mip_index_dir = {}

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

def model_test(test_loader):
    load_name = model_dir + 'best.pth'
    seg_model = get_segmentation_model(inchannel=1, load_name=load_name)
    seg_model.eval()
    # seg_model.train()
    dice_sum, dice_mip_sum = 0, 0
    out_path = model_dir + 'result/'
    os.makedirs(out_path, exist_ok=True)
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            print(data['name'])
            img_name = root_path + 'mra_data1/size_img/' + data['name'][0] + '.nii.gz'
            if not os.path.exists(img_name):
                img_name = img_name.replace('/data/', '/data_add/')
            img_ = sitk.ReadImage(img_name)
            img3d = data['img3d'].to(device).float()
            lbl3d = data['lbl3d'].to(device).float()
            mip_img0, mip_lbl0 = data['mip_img0'].to(device).float(), data['mip_lbl0'].to(device).float()
            mip_label = [mip_lbl0]
            mip_index0 = mip_index_dir[data['name'][0]]
            mip_index = [mip_index0]

            result3d, result3d_mip, result_mip = seg_model(img3d, mip_img0, None, None, mip_index)
            result_mip_0, mip_label_0 = result_mip[0], mip_label[0]
            result_mip_0 = torch.argmax(result_mip_0, dim=1).unsqueeze(1)
            result3d = torch.argmax(result3d, dim=1).unsqueeze(1)
            dice_one_mip = 2.0 * (((mip_label_0 == 1) & (result_mip_0 == 1)).sum()) / ((mip_label_0 == 1).sum() + (result_mip_0 == 1).sum())
            dice_mip_sum += dice_one_mip

            dice_one = 2.0 * (((lbl3d == 1) & (result3d == 1)).sum()) / ((lbl3d == 1).sum() + (result3d == 1).sum())
            dice_sum += dice_one
            print(data['name'], dice_one * 100, dice_one_mip * 100)
            result3d = result3d.squeeze().detach().cpu().numpy()
            result3d = sitk.GetImageFromArray(result3d)
            result3d.CopyInformation(img_)
            sitk.WriteImage(result3d, out_path + data['name'][0] + '_label.nii.gz')
    print(i + 1)
    print('average dice:{:.2f}, average mip dice:{:.2f}'.format(dice_sum / (i + 1) * 100, dice_mip_sum / (i + 1) * 100))

def main():
    print(args.num_mip)
    if args.all_fold == 0:
        fold_index = '' + '.txt'
    else:
        fold_index = '_fold' + str(args.all_fold) + '.txt'
    test_set = MipDataset2TwoChannelFold('data_txt/test' + fold_index, root_path + 'mra_data1/', feature_num=args.num_fea_list,
        rec_path='reconstruction_label2_one_new/', suffix='-')#[:-1] + '_new/'
    for i in range(0, test_set.__len__()):
        data = test_set.__getitem__(i)
        mip_index0 = data['mip_index0']
        for j in range(0, len(mip_index0)):
            mip_index0[j] = mip_index0[j].to('cuda').unsqueeze(0)
        mip_index_dir[data['name']] = mip_index0
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4)
    model_test(test_loader)

if __name__ == '__main__':
    args.num_mip = 1
    if args.all_fold == 0:
        fold_index = '/'
    else:
        fold_index = '_fold' + str(args.all_fold) + '/'
    model_dir_p = root_path + args.model_dir + 'two_channel' + fold_index
    model_dir = model_dir_p + 'ablation_CL_un_model-init-4/'
    os.makedirs(model_dir, exist_ok=True)
    main()