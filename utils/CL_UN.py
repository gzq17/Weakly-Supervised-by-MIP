from curses import echo
import os
import random
import argparse
import time
import numpy as np
import torch, sys, copy
import torch.nn.functional as F
from scipy.spatial.distance import cdist
torch.cuda.manual_seed(1234)
from network.mip_network import MipModelNum1TwoChannelDrop
from utils.dataset_mip import MipDataset2TwoChannelFold
from torch.utils.data import DataLoader
import SimpleITK as sitk
import cleanlab
import math
from scipy.ndimage.morphology import distance_transform_edt
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
root_path = '/disk1/guozhanqiang/Cerebral/mip_data/'

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', type=str, default='model/work_expansion/mra_data1_model/')#mip_model1
parser.add_argument('--num_mip', type=int, default=1)
parser.add_argument('--num_fea_list', type=float, default=[2, 6, 16, 32, 64, 128])
parser.add_argument('--drop', type=bool, default=True)
parser.add_argument('--all_fold', type=int, default=0)
args = parser.parse_args()
device = 'cuda'
tau = 0.05
if args.all_fold == 0:
    fold_index = '' + '.txt'
else:
    fold_index = '_fold' + str(args.all_fold) + '.txt'

def read_txt(file_name):
    if file_name is None:
        return None
    name_list = []
    f = open(file_name, 'r')
    a = f.readlines()
    for name in a:
        name_list.append(name[:-1])
    return name_list

def adjust_lr(optimizer, epoch, lr):
    lr_c = lr * ((1 - epoch/(args.epochs + 1)) ** 0.9)
    for p in optimizer.param_groups:
        p['lr'] = lr_c

def get_segmentation_model(inchannel, load_name=None):
    if load_name is None:
        seg_model = MipModelNum1TwoChannelDrop(inchannel, 2, activate='relu', norm='instance', num_list=args.num_fea_list[1:], mip_num=args.num_mip)
    else:
        seg_model = MipModelNum1TwoChannelDrop(inchannel, 2, activate='relu', norm='instance', num_list=args.num_fea_list[1:], mip_num=args.num_mip)
        seg_model.load_state_dict(torch.load(load_name))
    return seg_model.to(device)

def save_img(img, img_, name):
    img = sitk.GetImageFromArray(img)
    img.CopyInformation(img_)
    sitk.WriteImage(img, name)

def cl_module(flod_loader):
    out_path = root_path + 'mra_data1/reconstruction_label2_one_new_cl_un/'
    load_name = model_dir + 'best.pth'
    seg_model = get_segmentation_model(inchannel=1, load_name=load_name)
    print(load_name)
    os.makedirs(out_path, exist_ok=True)
    patch_size = [128, 384, 384]
    dice0_sum, dice1_sum, dice2_sum, k = 0, 0, 0, 0
    err1_sum, err2_sum, sum1_sum, sum2_sum = 0, 0, 0, 0
    d_th2, varepsilon2 = 0, 0
    for i, data in enumerate(flod_loader):
        out_path = root_path + 'mra_data1/reconstruction_label2_one_new_cl_un/'
        print(data['name'])
        k += 1
        img_name = root_path + 'mra_data1/size_img/' + data['name'][0] + '.nii.gz'
        center_name = root_path + 'mra_data1/reconstruction_label2_one_new/' + data['name'][0] + '_reconstruction_center.nii.gz'
        if not os.path.exists(img_name):
            img_name = img_name.replace('/data/', '/data_add/')
            center_name = center_name.replace('/data/', '/data_add/')
            out_path = out_path.replace('/data/', '/data_add/')
            os.makedirs(out_path, exist_ok=True)
        center_img = sitk.GetArrayFromImage(sitk.ReadImage(center_name))
        img_ = sitk.ReadImage(img_name)
        img3d = data['img3d'].to(device).float()
        lbl3d = data['lbl3d'].to(device).float()
        lbl3d_fore = data['lbl3d_fore'].to(device).float()
        lbl3d_ctr = data['lbl3d_ctr'].to(device).float()
        lbl3d_back = data['lbl3d_back'].to(device).float()
        mip_img0, mip_lbl0, mip_index0 = data['mip_img0'].to(device).float(), data['mip_lbl0'].to(device).float(), data['mip_index0']
        mip_label = [mip_lbl0]
        mip_index = [mip_index0]

        lbl3d_np = np.squeeze(lbl3d.cpu().detach().numpy())
        img3d_np = np.squeeze((copy.deepcopy(img3d)).cpu().detach().numpy())
        lbl3d_back_np = np.squeeze((copy.deepcopy(lbl3d_back)).cpu().detach().numpy())
        gray_tau = (img3d_np * lbl3d_np).sum() / (lbl3d_np.sum())

        mask_join = (img3d >= tau)
        un_label = ((lbl3d_fore==0) & (lbl3d_back==0))
        labeled = ((lbl3d_fore==1) | (lbl3d_back==1))
        mask_fuse = ((lbl3d_fore==1) & (lbl3d_back==1))
        lbl3d_fore_np = copy.deepcopy(lbl3d_fore.squeeze().cpu().detach().numpy())
        lbl3d_fore_dist = distance_transform_edt(1 - lbl3d_fore_np)
        print(lbl3d_fore.shape, lbl3d_back.shape)
        noisy_label_batch = lbl3d_fore.clone()
        noisy_label_batch[lbl3d_back == 1] = 0
        LQ_labeled_volume_batch = img3d
        with torch.no_grad():
            seg_model.eval()            
            ema_output_no_noise, result3d_mip, result_mip = seg_model(LQ_labeled_volume_batch, mip_img0, None, None, mip_index)
            ema_output_soft_no_noise = torch.softmax(ema_output_no_noise, dim=1)
            seg_model.train()
        # 1: tensor to npy
        mask_join = mask_join.cpu().detach().numpy()
        mask_fuse = mask_fuse.cpu().detach().numpy()
        un_label = un_label.cpu().detach().numpy()
        labeled = labeled.cpu().detach().numpy()

        masks_np = noisy_label_batch.cpu().detach().numpy()
        ema_output_soft_np = ema_output_soft_no_noise.cpu().detach().numpy()
        index_fuse = np.where(mask_fuse == 1)
        index_unlabel = np.where(un_label == 1)
        masks_np[index_fuse] = np.argmax(ema_output_soft_np, axis=1, keepdims=True)[index_fuse]
        masks_np[index_unlabel] = np.argmax(ema_output_soft_np, axis=1, keepdims=True)[index_unlabel]
        # mask_join = mask_join & labeled
        index_join = np.where(mask_join == 1)
        index_join = index_join[2] * patch_size[1] * patch_size[2] + index_join[3] * patch_size[2] + index_join[4]

        ema_output_soft_np_accumulated_0 = np.swapaxes(ema_output_soft_np, 1, 2)
        ema_output_soft_np_accumulated_1 = np.swapaxes(ema_output_soft_np_accumulated_0, 2, 3)
        ema_output_soft_np_accumulated_2 = np.swapaxes(ema_output_soft_np_accumulated_1, 3, 4)
        ema_output_soft_np_accumulated_3 = ema_output_soft_np_accumulated_2.reshape(-1, 2)
        ema_output_soft_np_accumulated = np.ascontiguousarray(ema_output_soft_np_accumulated_3)
        ema_output_soft_np_accumulated = ema_output_soft_np_accumulated[index_join]
        masks_np_accumulated = masks_np.reshape(-1).astype(np.uint8)
        noise_o = np.zeros(masks_np_accumulated.shape)
        masks_np_accumulated = masks_np_accumulated[index_join]
        assert masks_np_accumulated.shape[0] == ema_output_soft_np_accumulated.shape[0]
        CL_type = 'both'
        try:
            if CL_type in ['both']:
                noise = cleanlab.filter.find_label_issues(masks_np_accumulated, ema_output_soft_np_accumulated, filter_by='both', n_jobs=1)
            elif CL_type in ['prune_by_class', 'prune_by_noise_rate']:
                noise = cleanlab.filter.find_label_issues(masks_np_accumulated, ema_output_soft_np_accumulated, filter_by=CL_type, n_jobs=1)
            noise_o[index_join] = noise
            print(noise.sum())
            confident_maps_np = noise_o.reshape(-1, patch_size[0], patch_size[1], patch_size[2]).astype(np.uint8)
            
            # Correct the LQ label for our focused binary task
            correct_type = 'hard'
            if correct_type == 'hard':
                corrected_masks_np = masks_np + confident_maps_np * np.power(-1, masks_np)
                print('Smoothly correct the noisy label')
            if correct_type == 'smooth':
                smooth_arg = 0.8
                corrected_masks_np = masks_np + confident_maps_np * np.power(-1, masks_np) * smooth_arg
                print('Smoothly correct the noisy label')
            elif correct_type == 'uncertainty_smooth':
                T = 6
                _, _, d, w, h = LQ_labeled_volume_batch.shape
                volume_batch_r = LQ_labeled_volume_batch.repeat(2, 1, 1, 1, 1)
                stride = volume_batch_r.shape[0] // 2
                preds = torch.zeros([stride * T, 2, d, w, h]).cuda()
                seg_model.train()
                for i in range(T//2):
                    ema_inputs = volume_batch_r + torch.clamp(torch.randn_like(volume_batch_r) * 0.1, -0.2, 0.2)
                    with torch.no_grad():
                        ema_output_no_noise = seg_model(ema_inputs, mip_img0, None, None, mip_index, infer=True)
                        preds[2 * stride * i:2 * stride *(i + 1)] = ema_output_no_noise
                preds = torch.softmax(preds, dim=1)
                preds = preds.reshape(T, stride, 2, d, w, h)
                preds = torch.mean(preds, dim=0)
                uncertainty = -1.0 * torch.sum(preds*torch.log(preds + 1e-6), dim=1, keepdim=True)
                uncertainty = uncertainty/math.log(2) # normalize uncertainty to 0 to 1, cuz ln2 is the max value
                uncertainty_np = uncertainty.cpu().detach().numpy()
                uncertainty_np_squeeze = np.squeeze(uncertainty_np)
                smooth_arg = 1 - uncertainty_np_squeeze
                print(smooth_arg.shape)
                corrected_masks_np = masks_np + confident_maps_np * np.power(-1, masks_np) * smooth_arg
                print('Uncertainty-based smoothly correct the noisy label')
            else:
                corrected_masks_np = masks_np + confident_maps_np * np.power(-1, masks_np)
                print('Hard correct the noisy label')
            corrected_masks_np = labeled * corrected_masks_np
            corrected_masks_np_back = (1 - corrected_masks_np) * labeled

            corrected_masks_np[0, 0][center_img == 1] = 1
            corrected_masks_np_back[0, 0][center_img == 1] = 0
            noisy_label_batch = torch.from_numpy(corrected_masks_np).cuda(img3d.device.index)#.long()
            corrected_masks_back = torch.from_numpy(corrected_masks_np_back).cuda(img3d.device.index)
        except Exception as e:
                print('Cannot identify noises')
        corrected_masks_np_back[mask_fuse == 1] = 0
        # save_img(corrected_masks_np[0, 0], img_, out_path + data['name'][0] + '_reconstruction_label.nii.gz')
        save_img(corrected_masks_np_back[0, 0], img_, out_path + data['name'][0] + '_reconstruction_back.nii.gz')
        save_img(center_img, img_, out_path + data['name'][0] + '_reconstruction_center.nii.gz')
        masks_np = torch.from_numpy(masks_np).cuda(img3d.device.index)
        if correct_type == 'hard':
            err1 = ((lbl3d_fore == 1) & (lbl3d == 0)).sum() / (lbl3d_fore == 1).sum()
            err2 = ((noisy_label_batch == 1) & (lbl3d == 0)).sum() / (noisy_label_batch == 1).sum()
            err1_sum += err1
            err2_sum += err2
            sum1_sum += (lbl3d_fore == 1).sum()
            sum2_sum += (noisy_label_batch == 1).sum()
            print('err1:{:.3f}, err2:{:.3f}, ori sum:{}, new sum:{}'.format(err1, err2, (lbl3d_fore == 1).sum(), (noisy_label_batch == 1).sum()))
        
        dice0 = 2 * (masks_np * lbl3d).sum() / (masks_np.sum() + lbl3d.sum()) * 100
        dice1 = 2 * (lbl3d_fore * lbl3d).sum() / (lbl3d_fore.sum() + lbl3d.sum()) * 100
        dice2 = 2 * (noisy_label_batch * lbl3d).sum() / (noisy_label_batch.sum() + lbl3d.sum()) * 100
        dice0_sum += dice0
        dice1_sum += dice1
        dice2_sum += dice2
        print('argmax dice:{:.2f}, fore ori dice:{:.2f}, new dice:{:.2f}'.format(dice0, dice1, dice2))
    print('average argmax dice :{:.2f}, ori dice:{:.2f}, new dice:{:.2f}'.format(dice0_sum / k, dice1_sum / k, dice2_sum / k))
    print(err1_sum / k, err2_sum / k, sum1_sum / k, sum2_sum / k)

def uncertainty_module(flod_loader):
    out_path = root_path + 'mra_data1/reconstruction_label2_one_new_cl_un/'
    os.makedirs(out_path, exist_ok=True)
    load_name = model_dir + 'best.pth'
    seg_model = get_segmentation_model(inchannel=1, load_name=load_name)
    print(load_name)
    err_sum_list = [0, 0, 0]
    num_list = [0, 0, 0]
    dice0_sum, dice1_sum, dice2_sum, k = 0, 0, 0, 0
    d_th2, varepsilon2 = 4.0, 0.75
    for i, data in enumerate(flod_loader):
        print(data['name'])
        k += 1
        img_name = root_path + 'mra_data1/size_img/' + data['name'][0] + '.nii.gz'
        center_name = root_path + 'mra_data1/reconstruction_label2_one_new/' + data['name'][0] + '_reconstruction_center.nii.gz'
        if not os.path.exists(img_name):
            img_name = img_name.replace('/data/', '/data_add/')
            center_name = center_name.replace('/data/', '/data_add/')
            out_path = out_path.replace('/data/', '/data_add/')
            os.makedirs(out_path, exist_ok=True)
        center_img = sitk.GetArrayFromImage(sitk.ReadImage(center_name))
        img_ = sitk.ReadImage(img_name)
        img3d = data['img3d'].to(device).float()
        lbl3d = data['lbl3d'].to(device).float()
        lbl3d_fore = data['lbl3d_fore'].to(device).float()
        lbl3d_ctr = data['lbl3d_ctr'].to(device).float()
        lbl3d_back = data['lbl3d_back'].to(device).float()
        mip_img0, mip_lbl0, mip_index0 = data['mip_img0'].to(device).float(), data['mip_lbl0'].to(device).float(), data['mip_index0']
        mip_label = [mip_lbl0]
        mip_index = [mip_index0]
        un_label = ((lbl3d_fore==0) & (lbl3d_back==0))
        LQ_labeled_volume_batch = img3d

        with torch.no_grad():
            seg_model.eval()            
            ema_output_no_noise, result3d_mip, result_mip = seg_model(LQ_labeled_volume_batch, mip_img0, None, None, mip_index)
            ema_output_soft_no_noise = torch.softmax(ema_output_no_noise, dim=1)
            seg_model.train()
        ema_output_soft_np = ema_output_soft_no_noise.cpu().detach().numpy()
        un_label = un_label.cpu().detach().numpy()
        lbl3d_fore = np.squeeze(lbl3d_fore.cpu().detach().numpy())
        lbl3d_back = np.squeeze(lbl3d_back.cpu().detach().numpy())
        lbl3d = np.squeeze(lbl3d.cpu().detach().numpy())
        img3d_np = np.squeeze((copy.deepcopy(img3d)).cpu().detach().numpy())
        gray_tau = (img3d_np * lbl3d).sum() / (lbl3d.sum()) * 0.9
        cronary_tau = (img3d_np * lbl3d).sum() / (lbl3d.sum())
        fuse_mask = (lbl3d_back == 1) & (lbl3d_fore == 1)
        print((fuse_mask * (1 - lbl3d)).sum() / fuse_mask.sum(), fuse_mask.sum())

        T = 6
        _, _, d, w, h = LQ_labeled_volume_batch.shape
        volume_batch_r = LQ_labeled_volume_batch.repeat(2, 1, 1, 1, 1)
        stride = volume_batch_r.shape[0] // 2
        preds = torch.zeros([stride * T, 2, d, w, h]).cuda()
        seg_model.train()
        for i in range(T//2):
            ema_inputs = volume_batch_r + torch.clamp(torch.randn_like(volume_batch_r) * 0.1, -0.2, 0.2)
            with torch.no_grad():
                ema_output_no_noise = seg_model(ema_inputs, mip_img0, None, None, mip_index, infer=True)
                preds[2 * stride * i:2 * stride *(i + 1)] = ema_output_no_noise
        preds = torch.softmax(preds, dim=1)
        preds = preds.reshape(T, stride, 2, d, w, h)
        preds = torch.mean(preds, dim=0)
        uncertainty = -1.0 * torch.sum(preds*torch.log(preds + 1e-6), dim=1, keepdim=True)
        uncertainty = uncertainty/math.log(2) # normalize uncertainty to 0 to 1, cuz ln2 is the max value
        uncertainty_np = uncertainty.cpu().detach().numpy()
        uncertainty_np_squeeze = np.squeeze(uncertainty_np)
        un_label = np.squeeze(un_label)
        print(uncertainty_np_squeeze.mean(), (uncertainty_np_squeeze * un_label).sum() / un_label.sum())
        smooth_arg = 1 - uncertainty_np_squeeze 
        err = ((lbl3d_fore == 1) & (lbl3d == 0)).sum() / (lbl3d_fore.sum())
        re_ratio = lbl3d_fore.sum()# / lbl.sum()
        err2 = ((lbl3d_back == 1) & (lbl3d == 1)).sum() / (lbl3d_back.sum())
        re_ratio2 = lbl3d_back.sum()

        print('init fore num:{}, err:{:.3f}'.format(re_ratio, err * 100))
        print('init back num:{}, err:{:.3f}'.format(re_ratio2, err2 * 100))
        err_sum_list[0] += err * 100
        num_list[0] += re_ratio
        uncertainty_ave_unlabeled = (smooth_arg * un_label).sum() / un_label.sum()
        tau = 1.0
        new_mask = (smooth_arg * un_label) > (tau * uncertainty_ave_unlabeled)
        preds = preds.cpu().detach().numpy()
        pred_reuslt = np.argmax(ema_output_soft_np, axis=1, keepdims=True)
        pred_reuslt2 = np.argmax(preds, axis=1, keepdims=True)
        un_label = un_label[np.newaxis, np.newaxis, :, :, :]
        fuse_mask = fuse_mask[np.newaxis, np.newaxis, :, :, :]

        corrected_masks_np = np.zeros(un_label.shape)
        pred_certain = pred_reuslt == pred_reuslt2
        pred_un_label_mask = pred_certain & un_label
        new_mask_index = np.where(pred_un_label_mask == 1)
        corrected_masks_np_back = copy.deepcopy(corrected_masks_np) + 2

        corrected_masks_np[new_mask_index] = pred_reuslt[new_mask_index]
        corrected_masks_np = np.squeeze(corrected_masks_np)
        corrected_masks_np_back[new_mask_index] = pred_reuslt[new_mask_index]
        corrected_masks_np_back = np.squeeze(corrected_masks_np_back)
        corrected_un_ave = (smooth_arg * corrected_masks_np).sum() / corrected_masks_np.sum()

        un_list = []
        corrected_masks_np_index = np.where(corrected_masks_np == 1)
        for ii in range(0, corrected_masks_np_index[0].shape[0]):
            xx, yy, zz = corrected_masks_np_index[0][ii], corrected_masks_np_index[1][ii], corrected_masks_np_index[2][ii]
            un_list.append(smooth_arg[xx, yy, zz])
        un_list = sorted(un_list, reverse=True)
        aa = int(len(un_list) * 0.75)
        new_new_mask = smooth_arg <= un_list[aa]

        index_labeled_fore = np.where(lbl3d_fore == 1)
        index_labeled_fore = np.concatenate([np.array(index_labeled_fore[0])[:, np.newaxis],
                                            np.array(index_labeled_fore[1])[:, np.newaxis],
                                            np.array(index_labeled_fore[2])[:, np.newaxis]], axis=1)
        corrected_masks_np[new_new_mask] = 0
        corrected_masks_np_index = np.where(corrected_masks_np == 1)
        for ii in range(0, corrected_masks_np_index[0].shape[0]):
            xx, yy, zz = corrected_masks_np_index[0][ii], corrected_masks_np_index[1][ii], corrected_masks_np_index[2][ii]
            point_one = np.array([[xx, yy, zz]])
            dist = cdist(point_one, index_labeled_fore, metric='euclidean')
            if dist.min() > d_th2:
                corrected_masks_np[xx, yy, zz] = 0

        un_list = []
        corrected_masks_np_index = np.where(corrected_masks_np_back == 0)
        for ii in range(0, corrected_masks_np_index[0].shape[0]):
            xx, yy, zz = corrected_masks_np_index[0][ii], corrected_masks_np_index[1][ii], corrected_masks_np_index[2][ii]
            un_list.append(smooth_arg[xx, yy, zz])
        un_list = sorted(un_list, reverse=True)
        aa = int(len(un_list) * 0.45)
        new_new_mask = smooth_arg <= un_list[aa]
        corrected_masks_np_back[new_new_mask] = 1
        corrected_masks_np_index = np.where(corrected_masks_np_back == 0)
        for ii in range(0, corrected_masks_np_index[0].shape[0]):
            xx, yy, zz = corrected_masks_np_index[0][ii], corrected_masks_np_index[1][ii], corrected_masks_np_index[2][ii]
            # if img3d_np[xx, yy, zz] > cronary_tau * varepsilon2:#cronary 0.4
            #     corrected_masks_np_back[xx, yy, zz] = 1

        pred_reuslt = np.squeeze(pred_reuslt)
        pred_reuslt2 = np.squeeze(pred_reuslt2)
        dice = (pred_reuslt * lbl3d).sum() * 2 / (pred_reuslt.sum() + lbl3d.sum())
        dice0_sum += dice
        dice2 = (pred_reuslt2 * lbl3d).sum() * 2 / (pred_reuslt2.sum() + lbl3d.sum())
        dice1_sum += dice2
        fore_lbl_ = copy.deepcopy(lbl3d_fore)
        new_mask_index = np.where(un_label[0, 0] == 1)
        fore_lbl_[new_mask_index] = pred_reuslt[new_mask_index]
        back_lbl_ = 1 - fore_lbl_
        dice3 = (fore_lbl_ * lbl3d).sum() * 2 / (fore_lbl_.sum() + lbl3d.sum())
        dice2_sum += dice3
        err = ((corrected_masks_np == 1) & (lbl3d == 0)).sum() / (corrected_masks_np.sum())
        re_ratio = corrected_masks_np.sum()# / lbl.sum()
        err2 = ((corrected_masks_np_back == 0) & (lbl3d == 1)).sum() / ((corrected_masks_np_back == 0).sum())
        re_ratio2 = (corrected_masks_np_back == 0).sum()
        fore_lbl_ = copy.deepcopy(lbl3d_fore)
        fore_lbl_[fuse_mask[0,0] == 1] = 0
        fore_lbl_[corrected_masks_np == 1] = 1
        if err2 < 0.1:
            fore_lbl_[corrected_masks_np == 1] = 1
        print(fore_lbl_.sum())
        back_lbl_ = copy.deepcopy(lbl3d_back)
        back_lbl_[fuse_mask[0,0] == 1] = 0
        back_lbl_[corrected_masks_np_back == 0] = 1
        print('新增前景数量:{}，错误率:{:.3f}'.format(re_ratio, err * 100))
        print('新增背景数量:{}，错误率:{:.3f}'.format(re_ratio2, err2 * 100))
        save_img(fore_lbl_, img_, out_path + data['name'][0] + '_reconstruction_label.nii.gz')
        save_img(back_lbl_, img_, out_path + data['name'][0] + '_reconstruction_back.nii.gz')
        save_img(center_img, img_, out_path + data['name'][0] + '_reconstruction_center.nii.gz')
    print('drop dice:{}, dice:{},new dice:{}'.format(dice0_sum / k, dice1_sum / k, dice2_sum / k))

def main(img_b, img_e):
    print(args.num_mip)
    flod_set = MipDataset2TwoChannelFold('data_txt/train' + fold_index, root_path + 'mra_data1/', feature_num=args.num_fea_list,
        rec_path='reconstruction_label2_one_new/', suffix='-', img_b=img_b, img_e=img_e)#[:-1] + '_new/'
    flod_loader = DataLoader(flod_set, batch_size=1, shuffle=False, num_workers=4)
    uncertainty_module(flod_loader)
    cl_module(flod_loader)

def fuse_label_public_fore():
    out_path1 = root_path + 'mra_data1/reconstruction_label2_one_new_cl/'
    out_path2 = root_path + 'mra_data1/reconstruction_label2_one_new_un/'
    out_path = root_path + 'mra_data1/reconstruction_label2_one_new_cl_un/'
    lbl_path = root_path + 'mra_data1/size_lbl/'
    ori_path = root_path + 'mra_data1/reconstruction_label2_one_new/'
    test_list = read_txt(root_path + 'mra_data1/data_txt/train.txt')
    os.makedirs(out_path, exist_ok=True)
    for name in test_list:
        img_ = sitk.ReadImage(ori_path + name + '_reconstruction_label.nii.gz')
        ori_lbl = sitk.GetArrayFromImage(sitk.ReadImage(ori_path + name + '_reconstruction_label.nii.gz'))
        ori_back = sitk.GetArrayFromImage(sitk.ReadImage(ori_path +name + '_reconstruction_back.nii.gz'))
        lbl = sitk.GetArrayFromImage(sitk.ReadImage(lbl_path + name + '-label.nii.gz'))
        un_label = (ori_lbl == 0) & (ori_back == 0)
        labeled = (ori_lbl == 1) | (ori_back == 1)
        fuse_label = (ori_lbl == 1) & (ori_back == 1)
        
        img_name1 = out_path1 + name + '_reconstruction_label.nii.gz'
        img_name2 = out_path2 + name + '_reconstruction_label.nii.gz'
        out_name = out_path + name + '_reconstruction_label.nii.gz'
        img1 = sitk.GetArrayFromImage(sitk.ReadImage(img_name1))
        img2 = sitk.GetArrayFromImage(sitk.ReadImage(img_name2))

        new_img = copy.deepcopy(img1)
        new_img[(img2 * un_label) == 1] = 1
        # new_img[fuse_label == 1] = 0
        new_img_ = sitk.GetImageFromArray(new_img)
        new_img_.CopyInformation(img_)
        sitk.WriteImage(new_img_, out_name)

def fuse_label_public_back():
    out_path1 = root_path + 'mra_data1/reconstruction_label2_one_new_cl/'
    out_path2 = root_path + 'mra_data1/reconstruction_label2_one_new_un/'
    out_path = root_path + 'mra_data1/reconstruction_label2_one_new_cl_un/'
    lbl_path = root_path + 'mra_data1/size_lbl/'
    ori_path = root_path + 'mra_data1/reconstruction_label2_one_new/'
    test_list = read_txt(root_path + 'mra_data1/data_txt/train.txt')
    os.makedirs(out_path, exist_ok=True)
    for name in test_list:
        img_ = sitk.ReadImage(ori_path + name + '_reconstruction_back.nii.gz')
        ori_lbl = sitk.GetArrayFromImage(sitk.ReadImage(ori_path + name + '_reconstruction_back.nii.gz'))
        ori_back = sitk.GetArrayFromImage(sitk.ReadImage(ori_path +name + '_reconstruction_back.nii.gz'))
        lbl = sitk.GetArrayFromImage(sitk.ReadImage(lbl_path + name + '-label.nii.gz'))
        un_label = (ori_lbl == 0) & (ori_back == 0)
        labeled = (ori_lbl == 1) | (ori_back == 1)
        fuse_label = (ori_lbl == 1) & (ori_back == 1)
        
        img_name1 = out_path1 + name + '_reconstruction_back.nii.gz'
        img_name2 = out_path2 + name + '_reconstruction_back.nii.gz'
        out_name = out_path + name + '_reconstruction_back.nii.gz'
        img1 = sitk.GetArrayFromImage(sitk.ReadImage(img_name1))
        img2 = sitk.GetArrayFromImage(sitk.ReadImage(img_name2))

        new_img = copy.deepcopy(img1)
        new_img[(img2 * un_label) == 1] = 1
        new_img_ = sitk.GetImageFromArray(new_img)
        new_img_.CopyInformation(img_)
        sitk.WriteImage(new_img_, out_name)

if __name__ == '__main__':
    for ii in range(0, 3):
        fold = str(ii)
        if fold == '0':
            img_b, img_e = 20, 30
        elif fold == '1':
            img_b, img_e = 10, 20
        elif fold == '2':
            img_b, img_e = 0, 10
        else:
            exit()
        print(fold, img_b, img_e)
        model_dir_p = root_path + args.model_dir + 'two_channel' + fold_index[:-4] + '/'
        if args.drop:
            model_dir = model_dir_p + 'one_model1_drop_fold' + fold + '/'
        else:
            model_dir = model_dir_p + 'CL_model2_test1/'
        print(model_dir)
        main(img_b, img_e)
    #optional
    #fuse_label_public_fore()
    #fuse_label_public_back()
