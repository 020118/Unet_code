import torch
import torch.nn as nn
import math
from functools import partial
import torch.nn.functional as F


def weight_init(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, init_gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)
    print('initialize network with %s type' % init_type)
    net.apply(init_func)


def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio=0.05, warmup_lr_ratio=0.1, no_aug_iters_ratio=0.05, step_num=10):
    def yolox_warmup_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_start_lr, no_aug_iters, iters):
        if iters <= warmup_total_iters:
            lr = (lr - warmup_start_lr) * pow(iters/float(warmup_total_iters), 2) + warmup_start_lr
        elif iters >= total_iters - no_aug_iters:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                1.0 + math.cos(math.pi*(iters-warmup_total_iters)/(iters-warmup_total_iters-no_aug_iters))
                )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1")
        n = iters // step_size
        out_lr = lr * decay_rate ** n
        return out_lr
    
    if lr_decay_type == 'cos':
        warmup_total_iters = min(max(total_iters*warmup_iters_ratio, 1), 3)
        warmup_start_lr = max(lr * warmup_lr_ratio, 1e-6)
        no_aug_iters = min(max(total_iters * no_aug_iters_ratio, 1), 15)
        func = partial(yolox_warmup_cos_lr, lr, min_lr, total_iters, warmup_total_iters, warmup_start_lr, no_aug_iters)
    else:
        decay_rate = (min_lr / lr) ** (1 / (step_num - 1))
        step_size = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)
    return func


def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def Focal_Loss(inputs, targets, cls_weights, num_classes=21, alpha=0.5, gamma=2):
    n, c, h, w = inputs.size()
    nt, ht, wt = targets.size()

    if h != ht or w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode='bilinear', align_corners=True)

    temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    temp_targets = targets.view(-1)

    #CrossEntropyLoss返回的是-logp，故下面logpt = -nn.CrossEntropyLoss
    logpt = -nn.CrossEntropyLoss(weight=cls_weights, ignore_index=num_classes, reduction='none')(temp_inputs, temp_targets)
    pt = torch.exp(logpt)
    if alpha is not None:
        logpt *= alpha
    loss = -((1 - pt) ** gamma) * logpt
    loss = loss.mean()
    return loss


def CE_Loss(inputs, targets, cls_weights, num_classes=21):
    n, c, h, w = inputs.size()
    nt, ht, wt = targets.size()

    if h != ht or w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode='bilinear', align_corners=True)
    
    temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    temp_targets = targets.view(-1)
    
    loss = nn.CrossEntropyLoss(weight=cls_weights, ignore_index=num_classes)(temp_inputs, temp_targets)
    return loss


def Dice_Loss(inputs, targets, beta=1, smooth=1e-5):
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = targets.size()

    if h != ht or w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode='bilinear', align_corners=True)

    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c), -1)
    temp_targets = targets.view(n, -1, ct)

    tp = torch.sum(temp_targets[...,:-1] * temp_inputs, dim=[0, 1])
    fp = torch.sum(temp_inputs, dim=[0, 1]) - tp
    fn = torch.sum(temp_targets[...,:-1], dim=[0, 1]) - tp

    score = ((1 + beta**2) * tp + smooth) / ((1 + beta**2) * tp + beta**2 * fn + fp + smooth)
    dice_loss = 1 - torch.mean(score)
    return dice_loss





