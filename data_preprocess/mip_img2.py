import cv2
import SimpleITK as sitk
import numpy as np
import copy, os, sys
from tqdm import tqdm
root_path = '/disk1/guozhanqiang/Cerebral/mip_data/coronary_data/'

def save_mip():
    img_path = root_path + 'size_img/'
    lbl_path = root_path + 'size_lbl/'
    remove_path = root_path + 'check_img/'
    out_path = root_path + 'mip_img2/'
    name_list = sorted(os.listdir(img_path))
    for name in name_list:
        name = name[:-7]
        print(name)
        img_name = img_path + name + '.nii.gz'
        lbl_name = lbl_path + name + '-label.nii.gz'
        remove_name = remove_path + name + '_remove_label.nii.gz'
        img_ = sitk.ReadImage(lbl_name)
        lbl = sitk.GetArrayFromImage(img_)
        img = sitk.GetArrayFromImage(sitk.ReadImage(img_name))
        other = sitk.GetArrayFromImage(sitk.ReadImage(remove_name))
        img[other == 1] = img.min()
        img = (img - img.min()) / (img.max() - img.min()) * 255.0
        for ii in range(0, 1):
            # print(name)
            img_slicer = np.max(img, ii)
            img_index = np.argmax(img, ii)
            lbl_index = np.expand_dims(img_index, ii)
            np.save(out_path + name + '_' + str(ii) + '.npy', img_index)
            cv2.imwrite(out_path + name + '_' + str(ii) + '.png', img_slicer)

if __name__ == '__main__':
    save_mip()
