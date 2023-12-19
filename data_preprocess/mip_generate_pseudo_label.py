import cv2
import SimpleITK as sitk
import numpy as np
import copy, os, sys
from skimage import transform
import skimage
from scipy.spatial.distance import cdist
from tqdm import tqdm
import torch
root_path = '/disk1/guozhanqiang/Cerebral/mip_data/data/'
# root_path = 'D:/research_work/Cerebral/data/weakly_supervised/cerebral_mra/data1/'

def read_txt(file_name):
    if file_name is None:
        return None
    name_list = []
    f = open(file_name, 'r')
    a = f.readlines()
    for name in a:
        name_list.append(name[:-1])
    return name_list

def center_reconstruction_one2(name, img, lbl, mip_num=1):
    mip_path = root_path + 'mip_img2/'
    mask_list = []
    for index in range(0, mip_num):
        lbl0 = cv2.imread(mip_path + name + '_' + str(index) + '_label.png', cv2.IMREAD_GRAYSCALE)
        lbl0[lbl0 > 0.5] = 1
        lbl0[lbl0 <= 0.5] = 0
        center0 = lbl0
        # center0 = skeletonize(lbl0)
        center0 = torch.from_numpy(center0).unsqueeze(index).float()
        mip_index = np.load(mip_path + name + '_' + str(index) + '.npy')
        mip_index = torch.from_numpy(mip_index).unsqueeze(index).long()
        mask = torch.zeros(img.shape).float()
        mask = mask.scatter(index, mip_index, center0)
        mask = mask.numpy()
        mask_list.append(mask)
    if len(mask_list) == 1:
        mask = (mask_list[0] > 0.5)
    elif len(mask_list) == 2:
        mask = (mask_list[0] > 0.5) | (mask_list[1] > 0.5)
    else:
        mask = (mask_list[0] > 0.5) | (mask_list[1] > 0.5) | (mask_list[2] > 0.5)
    print('构造的中心线出错的比例以及一共的数量:', (((mask == 1) & (lbl == 0))).sum(), (mask == 1).sum())
    re_center = np.zeros(lbl.shape)
    re_center[mask] = 1
    return re_center

def grow_tree_one(img, lbl, points_img, grey_th=0.9):
    points = np.where(points_img == 1)
    points = np.concatenate([np.array(points[0])[:, np.newaxis],
                                    np.array(points[1])[:, np.newaxis],
                                    np.array(points[2])[:, np.newaxis]], axis=1)
    ####data3(data) 选择的是0.8  data1(mra_data1)选择的是0.85  cta_data选择的是0.9
    # print((points_img * img).sum() / points_img.sum() * grey_th)
    th_img = img > ((points_img * img).sum() / points_img.sum() * grey_th)
    # th_img2 = img < ((points_img * img).sum() / points_img.sum() * 1.2)
    # th_img = th_img & th_img2
    th_img = th_img.astype(np.uint8)
    [label, num] = skimage.measure.label(th_img, return_num=True)
    new_new_center = np.zeros(img.shape)
    for i in tqdm(range(1, num + 1)):
        temp = (label == i).sum()
        if temp < 10:#MRA 都是10
            continue
        one_points = np.where(label == i)
        one_points = np.concatenate([np.array(one_points[0])[:, np.newaxis],
                                 np.array(one_points[1])[:, np.newaxis],
                                 np.array(one_points[2])[:, np.newaxis]], axis=1)
        if cdist(one_points, points, metric='euclidean').min() == 0:
            new_new_center[label == i] = 1
    new_new_center[points_img == 1] = 1
    err = ((new_new_center == 1) & (lbl == 0)).sum() / new_new_center.sum()
    radio = (new_new_center == 1).sum() / lbl.sum()
    print('reconstruction label:{}, {:.3f}, {:.3f}'.format(new_new_center.sum(), err, radio))
    return new_new_center

def reconstruction_back2(num_mip=1):
    img_path = root_path + 'size_img/'
    lbl_path = root_path + 'size_lbl/'
    rec_path = root_path + 'reconstruction_label2_one_new/'
    mip_path = root_path + 'mip_img2/'
    name_list = sorted(os.listdir(img_path))
    ratio_sum, err_sum, k = 0, 0, 0
    for name in name_list:
        if name[-7:] != '.nii.gz':
            continue
        name = name[:-7]
        print(name)
        k += 1
        img_name = img_path + name + '.nii.gz'
        lbl_name = lbl_path + name + '_label.nii.gz'
        rec_center_name = rec_path + name + '_reconstruction_center.nii.gz'
        img_ = sitk.ReadImage(lbl_name)
        lbl = sitk.GetArrayFromImage(img_)
        img = sitk.GetArrayFromImage(sitk.ReadImage(img_name))
        rec_center = sitk.GetArrayFromImage(sitk.ReadImage(rec_center_name))
        ave_grey = (rec_center * img).sum() / rec_center.sum()
        mip_list = []
        for ii in range(0, num_mip):
            mip_0 = cv2.imread(mip_path + name + '_' + str(ii) + '_label.png', cv2.IMREAD_GRAYSCALE)
            mip_0[mip_0 > 0] = 1
            mip_list.append(mip_0)
        mask0 = np.zeros(img.shape)
        for i in range(0, mask0.shape[0]):
            mask0[i, :, :] = copy.deepcopy(1 - mip_list[0])
        if num_mip == 1:
            mask = (mask0 > 0)
        elif num_mip == 2:
            mask1 = np.zeros(img.shape)
            for i in range(0, mask1.shape[1]):
                mask1[:, i, :] = copy.deepcopy(1 - mip_list[1])
            mask = (mask0 > 0) | (mask1 > 0)
        else:
            mask1 = np.zeros(img.shape)
            mask2 = np.zeros(img.shape)
            for i in range(0, mask1.shape[1]):
                mask1[:, i, :] = copy.deepcopy(1 - mip_list[1])
            for i in range(0, mask2.shape[2]):
                mask2[:, :, i] = copy.deepcopy(1 - mip_list[2])
            mask = (mask0 > 0) | (mask1 > 0) | (mask2 > 0)
        ratio = (mask == 1).sum() / (lbl == 0).sum()
        ratio_sum += ratio
        err = ((mask == 1) & (lbl == 1)).sum() / (mask == 1).sum()
        err_sum += err
        print(ratio, err)
        back_mask = np.zeros(img.shape)
        back_mask[mask] = 1
        back_mask[img > ave_grey * 1.2] = 0
        back_mask[img < ave_grey * 0.2] = 1
        back_mask_ = sitk.GetImageFromArray(back_mask)
        back_mask_.CopyInformation(img_)
        sitk.WriteImage(back_mask_, rec_path + name + '_reconstruction_back.nii.gz')
    print(ratio_sum / k, err_sum / k)

def main_according_mip2(num_mip=1):
    img_path = root_path + 'size_img/'
    lbl_path = root_path + 'size_lbl/'
    out_path = root_path + 'reconstruction_label2_one_new/'
    os.makedirs(out_path, exist_ok=True)
    name_list = sorted(os.listdir(img_path))
    k = 1
    for name in name_list:
        if name[-7:] != '.nii.gz':
            continue
        print(name[:-7], '{}/{}'.format(k, len(name_list)))
        k += 1
        name = name[:-7]
        img_name = img_path + name + '.nii.gz'
        lbl_name = lbl_path + name + '-label.nii.gz'
        out_name = out_path + name + '_reconstruction_label.nii.gz'
        img_ = sitk.ReadImage(lbl_name)
        lbl = sitk.GetArrayFromImage(img_)
        img = sitk.GetArrayFromImage(sitk.ReadImage(img_name))
        high_low = img.max() - img.min()
        intensity_low = img.min() + high_low * 0.025
        intensity_high = img.max() - high_low * 0.025
        img = (img - intensity_low) / (intensity_high - intensity_low)
        img = np.clip(img, 0.0, 1.0)
        re_center = center_reconstruction_one2(name, img, lbl, mip_num=num_mip)
        re_center_ = sitk.GetImageFromArray(re_center)
        re_center_.CopyInformation(img_)
        sitk.WriteImage(re_center_, out_name.replace('_label.nii.gz', '_center.nii.gz'))
        reconstruction_label = grow_tree_one(img, lbl, re_center, grey_th=1.0)
        reconstruction_label = sitk.GetImageFromArray(reconstruction_label)
        reconstruction_label.CopyInformation(img_)
        sitk.WriteImage(reconstruction_label, out_name)

if __name__ == '__main__':
    main_according_mip2(num_mip=1)
    reconstruction_back2(num_mip=1)
