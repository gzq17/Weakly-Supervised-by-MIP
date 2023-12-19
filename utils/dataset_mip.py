import SimpleITK as sitk
import numpy as np
from torch.utils.data import Dataset
import cv2
from scipy.ndimage import rotate, zoom
import torch
import copy, math
import torch
import os
import torch.nn as nn
from torch.nn import functional as F 
device = 'cuda'
os.environ['CUDA_VISIBLE_DEVICES'] = '7'

def read_txt(file_name):
    if file_name is None:
        return None
    name_list = []
    f = open(file_name, 'r')
    a = f.readlines()
    for name in a:
        name_list.append(name[:-1])
    return name_list

class MipDataset2TwoChannelFold(Dataset):

    def __init__(self, img_name, path_parent, img3d_path='size_img/', lbl3d_path='size_lbl/',
     mip_path='mip_img2/', rec_path='reconstruction_label2/', feature_num=[1, 8, 16, 32, 64, 128],
     img_b=None, img_e=None, suffix='_', fold=None, add=None):
        if fold == 0:
            print('[0:20]')
            name_list = read_txt(path_parent + img_name)[0:20]
        elif fold == 1:
            print('[0:10][20:30]')
            name_list = read_txt(path_parent + img_name)[0:10] + read_txt(path_parent + img_name)[20:30]
        elif fold == 2:
            print('[10:30]')
            name_list = read_txt(path_parent + img_name)[10:30]
        else:
            if img_b is None:
                name_list = read_txt(path_parent + img_name)
            else:
                name_list = read_txt(path_parent + img_name)[img_b:img_e]
        self.path_parent = path_parent
        self.mip_path = mip_path
        self.feature_num = feature_num
        self.data_list = []
        print(path_parent + rec_path)
        print(path_parent + lbl3d_path)
        for name in name_list:
            print(name)
            one_path_parent = path_parent
            if len(name) < 10 and add is not None:
                one_path_parent = path_parent.replace('/data/', '/data_add/')
            img3d_name = one_path_parent + img3d_path + name + '.nii.gz'
            lbl3d_name = one_path_parent + lbl3d_path + name + suffix + 'label.nii.gz'
            lbl3d_fore_name = one_path_parent + rec_path + name + '_reconstruction_label.nii.gz'
            lbl3d_ctr_name = one_path_parent + rec_path + name + '_reconstruction_center.nii.gz'
            lbl3d_back_name = one_path_parent + rec_path + name + '_reconstruction_back.nii.gz'
            self.data_list.append([img3d_name, lbl3d_name, lbl3d_fore_name, lbl3d_ctr_name, lbl3d_back_name, name, one_path_parent])
    
    def __getitem__(self, index):
        data = self.data_list[index]
        img3d = np.load(data[0].replace('.nii.gz', '.npy'), mmap_mode='r')
        img3d = (img3d - img3d.min()) / (img3d.max() - img3d.min())
        # img3d = torch.from_numpy(img3d[np.newaxis, :, :, :])
        lbl3d = np.load(data[1].replace('.nii.gz', '.npy'), mmap_mode='r')
        lbl3d_fore = np.load(data[2].replace('.nii.gz', '.npy'), mmap_mode='r')
        # print(2 * (lbl3d * lbl3d_fore).sum() / (lbl3d.sum() + lbl3d_fore.sum()))
        lbl3d_ctr = np.load(data[3].replace('.nii.gz', '.npy'), mmap_mode='r')
        lbl3d_back = np.load(data[4].replace('.nii.gz', '.npy'), mmap_mode='r')
        mip_img0, mip_lbl0, mip_index0 = self.mip_data(0, data[-1], self.mip_path, data[5], self.feature_num)
        data_torch = {'img3d':torch.from_numpy(img3d[np.newaxis, :, :, :]),
                      'lbl3d':torch.from_numpy(lbl3d[np.newaxis, :, :, :]),
                      'lbl3d_fore':torch.from_numpy(lbl3d_fore[np.newaxis, :, :, :]),
                      'lbl3d_ctr':torch.from_numpy(lbl3d_ctr[np.newaxis, :, :, :]),
                      'lbl3d_back':torch.from_numpy(lbl3d_back[np.newaxis, :, :, :]),
                      'mip_img0':torch.from_numpy(mip_img0[np.newaxis, :, :]),
                      'mip_lbl0':torch.from_numpy(mip_lbl0[np.newaxis, :, :]),
                      'mip_index0':mip_index0,
                    #   'mip_index2':mip_index2,
                      'name':data[5]}
        return data_torch

    def mip_data(self, index, path_parent, mip_path, name, feature_num):
        mip_img = cv2.imread(path_parent + mip_path + name +  '_' + str(index) + '.png', cv2.IMREAD_GRAYSCALE)
        mip_img = (mip_img - mip_img.min()) / (mip_img.max() - mip_img.min())
        mip_lbl = cv2.imread(path_parent + mip_path + name +   '_' + str(index) + '_label.png', cv2.IMREAD_GRAYSCALE)
        mip_lbl[mip_lbl > 0] = 1.0
        mip_index = np.load(path_parent + mip_path + name +   '_' + str(index) + '.npy')
        index_list = []
        k = 0
        for num in feature_num:
            if k == 0 or k == 1:
                new_mip_index = copy.deepcopy(mip_index)
            else:
                ga = math.pow(2, k - 1)
                new_mip_index = zoom(copy.deepcopy(mip_index), (1 / ga, 1 / ga), order=1).astype(np.int64)
                new_mip_index = (new_mip_index // ga).astype(np.int64)
            k += 1
            index_torch = torch.zeros((num, *new_mip_index.shape))
            for i in range(0, num):
                index_torch[i, :, :] = torch.from_numpy(copy.deepcopy(new_mip_index))
            index_torch = index_torch.long().unsqueeze(index - 3)
            index_list.append(index_torch)
        return mip_img, mip_lbl, index_list

    def __len__(self):
        return len(self.data_list)
