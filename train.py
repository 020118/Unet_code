import torch
import numpy as np
import torch.distributed as dist
import os
import datetime

import torch.utils.data.distributed
import torch.utils.data.distributed
from utils import *
from nets.unet import unet
from nets.unet_training import weight_init, get_lr_scheduler, set_optimizer_lr
import torch.backends.cudnn as cudnn
import torch.optim as optim
from dataloader import UnetDataset, unet_dataset_collate
from torch.utils.data import DataLoader
from functools import partial
from callbacks import EvalCallback
from utils_fit import fit_one_epoch

if __name__ == "__main__":
    print(torch.__version__)
    print(torch.version.cuda)
    print(torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前使用的设备是：{device}")
    if device.type == "cuda":
        print("cuda device name:", torch.cuda.get_device_name(0))

    seed = 11
    distributed = False
    num_classes = 21
    backbone = "vgg"
    pretrained = False
    model_path = "model_data/unet_vgg_voc.pth"
    input_shape = [512, 512]

    #冻结阶段epoch参数
    Init_epoch = 0
    Freeze_epoch = 15
    Freeze_batchsize = 8

    #解冻阶段epoch参数
    Unfreeze_epoch = 30
    Unfreeze_batchsize = 4

    Freeze_train = True  

    Init_lr = 1e-4
    Min_lr = Init_lr * 0.01

    optimizer_type = 'adam'
    momentum = 0.9
    weight_decay = 0
    lr_decay_type = 'cos'
    saved_period = 5
    save_dir = 'logs'

    eval_flag = True
    eval_period = 5

    VOCdevkit_path = 'VOCdevkit'

    dice_loss = False
    focal_loss = False

    cls_weights = np.ones([num_classes], np.float32)

    nums_work = 4

    fp16 = True
    sync_bn = False
    cuda = True

    seed_everything(seed)

    #设置使用的显卡，根据是否采用分布式训练设置
    ngpus_per_node = torch.cuda.device_count()
    if distributed:
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        device = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("gpu device counts:", ngpus_per_node)
    else:
        local_rank = 0
        rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #下载预训练权重
    if pretrained:
        if distributed:
            if local_rank == 0:
                download_weight(backbone)
            dist.barrier()
        else:
            download_weight(backbone)
    
    model = unet(num_classes=num_classes, pretrained=pretrained, backbone=backbone).train()
    if not pretrained:
        weight_init(model)
    if model_path != '':
        if local_rank == 0:
            print('load weights: {}'.format(model_path))
        #根据预训练的key和模型的key进行加载
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        
        load_key, temp_dict, no_load_key = [], {}, []
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(v) == np.shape(model_dict[k]):
                load_key.append(k)
                temp_dict[k] = v
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)

        if local_rank == 0:
            print("\nSuccessful load key:", str(load_key)[:500], "...\nSuccessful load key num:", len(load_key))
            print("\nFail to load key:", str(no_load_key)[:500], "...\nFail to load key num:", len(no_load_key))
            print("\n\033[1;33;44m温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。\033[0m")
    
    if local_rank == 0:
        time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
        log_dir = os.path.join(save_dir, 'loss' + str(time_str))
        loss_history = Losshistory(log_dir, model, input_shape)
    else:
        loss_history = None

    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None
    
    model_train = model.train()  #开启训练

    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif sync_bn:
        print("sync is not support in one gpu or not distributed")

    if cuda:    
        if distributed:
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank], find_unused_parameters=True)
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()
    
    with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/train.txt"), "r") as f:
        train_lines = f.readlines()
    with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"), "r") as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)

    if local_rank == 0:
        show_config(
            num_classes = num_classes, backbone = backbone, model_path = model_path, input_shape = input_shape,
            Init_Epoch = Init_epoch, Freeze_Epoch = Freeze_epoch, UnFreeze_Epoch = Unfreeze_epoch, Freeze_batch_size = Freeze_batchsize, Unfreeze_batch_size = Unfreeze_batchsize, Freeze_Train = Freeze_train,
            Init_lr = Init_lr, Min_lr = Min_lr, optimizer_type = optimizer_type, momentum = momentum, lr_decay_type = lr_decay_type,
            save_period = saved_period, save_dir = save_dir, num_workers = nums_work, num_train = num_train, num_val = num_val
        )

    if True:
        Unfreeze_flag = False

        #冻结一定部分训练
        if Freeze_train:
            model.freeze_backbone()
        
        #如果不冻结训练，设置batch_size为unfreeze_batch_size
        batch_size = Freeze_batchsize if Freeze_train else Unfreeze_batchsize

        #判断当前batch_size，自适应调整学习率
        nbs = 16  #nominal batch size
        lr_limit_max = 1e-4 if optimizer_type == 'adam' else 1e-1
        lr_limit_min = 1e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit = min(max(batch_size/nbs*Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit = min(max(batch_size/nbs*Init_lr, lr_limit_min*1e-2), lr_limit_max*1e-2)

        #根据optimizer_type选择优化器
        optimizer = {
            'adam' : optim.Adam(model.parameters(), Init_lr_fit, betas=(momentum, 0.999), weight_decay=weight_decay),
            'sgd' : optim.SGD(model.parameters(), Init_lr_fit, momentum=momentum, nesterov=True, weight_decay=weight_decay)
        }[optimizer_type]

        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, Unfreeze_epoch)

        #判断每个epoch的长度
        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集")
        
        train_dataset = UnetDataset(train_lines, input_shape, num_classes, True, VOCdevkit_path)
        val_dataset = UnetDataset(val_lines, input_shape, num_classes, False, VOCdevkit_path)

        if distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
            batch_size = batch_size // ngpus_per_node
            shuffle = False
        else:
            train_sampler = None
            val_sampler = None
            shuffle = True
        
        gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=nums_work, drop_last=True, 
                               pin_memory=True, collate_fn=unet_dataset_collate, sampler=train_sampler,
                               worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
        gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=nums_work, drop_last=True,
                             pin_memory=True, collate_fn=unet_dataset_collate, sampler=val_sampler,
                             worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
        
        # 记录eval过程中的map曲线
        if local_rank == 0:
            eval_callback = EvalCallback(model, input_shape, num_classes, val_lines, VOCdevkit_path, log_dir,
                                         cuda, eval_flag=eval_flag, period=eval_period)
        else:
            eval_callback = None
        
        # 开始模型训练
        for epoch in range(Init_epoch, Unfreeze_epoch):
            # 如果模型有冻结部分，则解冻并设置参数
            if epoch >= Freeze_epoch and not Unfreeze_flag and Freeze_train:
                # 判断当前batch_size，自适应调整学习率
                nbs = 16
                lr_limit_max = 1e-4 if optimizer_type == 'adam' else 1e-1
                lr_limit_min = 1e-4 if optimizer_type == 'adam' else 5e-4
                Init_lr_fit = min(max(batch_size/nbs*Init_lr, lr_limit_min), lr_limit_max)
                Min_lr_fit = min(max(batch_size/nbs*Min_lr, lr_limit_min*1e-2), lr_limit_max*1e-2)

                # 获得学习率下降公式
                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, Unfreeze_epoch)

                model.unfreeze_backbone()

                epoch_step = num_train // batch_size
                epoch_step_val = num_val // batch_size
                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("数据集数量不够，请扩充数据集。")
                
                if distributed:
                    batch_size = batch_size // ngpus_per_node
                
                gen = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, sampler=train_sampler, 
                                 num_workers=nums_work, collate_fn=unet_dataset_collate, drop_last=True, 
                                 pin_memory=True, worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
                gen_val = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, num_workers=nums_work,
                                     drop_last=True, collate_fn=unet_dataset_collate, sampler=val_sampler, 
                                     pin_memory=True, worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
                
                Unfreeze_flag = True
            
            if distributed:
                train_sampler.set_epoch(epoch)
            
            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
            
            fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val,
                          gen, gen_val, Unfreeze_epoch, cuda, dice_loss, focal_loss, cls_weights, num_classes, fp16, 
                          scaler, saved_period, save_dir)
            
            if distributed:
                dist.barrier()

        if local_rank == 0:
            loss_history.writer.close()

            

