import torch
from torch import nn

def WeightedCorssLoss3dTwoChannel(y_fore, y_ctr, y_back, y_pred):
    y_pred = torch.clamp(y_pred, min=1e-5, max=1-1e-5)
    y_pred_back = y_pred[:, :1, ...]
    y_pred_fore = y_pred[:, 1:, ...]
    loss_fore = 1.0 / (torch.sum(y_fore)) * torch.sum(y_fore * torch.log(y_pred_fore))
    loss_ctr = 1.0 / (torch.sum(y_ctr)) * torch.sum(y_ctr * torch.log(y_pred_fore))
    # print((1 - y_pred).min())
    loss_back = 1.0 / (torch.sum(y_back)) * torch.sum(y_back * torch.log(y_pred_back))
    return -loss_fore, -loss_ctr, -loss_back

def dice_loss(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss

def one_hot_encoder(input_tensor):
    tensor_list = []
    for i in range(2):
        temp_prob = input_tensor == i * torch.ones_like(input_tensor)
        tensor_list.append(temp_prob)
    output_tensor = torch.cat(tensor_list, dim=1)
    return output_tensor.float()

def Dice2DTwoChannel(y_true_list, y_pred_list, num=3):
    loss_list = []
    for i in range(0, num):
        y_true, y_pred = y_true_list[i], y_pred_list[i]
        y_true_one = one_hot_encoder(y_true)
        dice1 = dice_loss(y_pred[:, 0, ...], y_true_one[:, 0, ...])
        dice2 = dice_loss(y_pred[:, 1, ...], y_true_one[:, 1, ...])
        loss_list.append(dice1 + dice2)
    if num == 1:
        return loss_list[0]
    elif num ==2:
        return loss_list[0] + loss_list[1]
    else:
        return loss_list[0] + loss_list[1] + loss_list[2]