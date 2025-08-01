import torch
import random
import numpy as np
import os
import scipy.signal
from torch.hub import load_state_dict_from_url
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
from PIL import Image

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def download_weight(backbone, model_dir="./model_data"):
    down_urls = {
        "vgg": "https://download.pytorch.org/models/vgg16-397923af.pth",
        "resnet50": "https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth"
    }
    url = down_urls[backbone]

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    load_state_dict_from_url(url, model_dir)


class Losshistory():
    def __init__(self, log_dir, model, input_shape, val_loss_flag=True):
        self.log_dir = log_dir
        self.val_loss_flag = val_loss_flag

        self.losses = []
        if self.val_loss_flag:
            self.val_loss = []
        
        os.makedirs(self.log_dir)
        self.writer = SummaryWriter(self.log_dir)
        try:
            dummy_input = torch.randn(2, 3, input_shape[0], input_shape[1])
            self.writer.add_graph(model, dummy_input)
        except:
            pass

    def append_loss(self, epoch, loss, val_loss=None):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.losses.append(loss)
        if self.val_loss_flag:
            self.val_loss.append(val_loss)
        
        with open(os.path.join(self.log_dir, "epoch_loss.txt"), 'a') as f:
            f.write(str(loss))
            f.write("\n")
        if self.val_loss_flag:
            with open(os.path.join(self.log_dir, "epoch_val_loss.txt"), 'a') as f:
                f.write(str(val_loss))
                f.write("\n")

        self.writer.add_scalar('loss', loss, epoch)
        if self.val_loss_flag:
            self.writer.add_scalar('val_loss', val_loss, epoch)
        
        self.loss_plot()

    def loss_plot(self):
        iters = range(len(self.losses))

        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth=2, label='train loss')
        if self.val_loss_flag:
            plt.plot(iter, self.val_loss, 'coral', linewidth=2, label='val loss')
        
        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15
            
            plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle='--', linewidth=2, label='smooth train loss')
            if self.val_loss_flag:
                plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle='--', linewidth=2, label='smooth val loss')
        except:
            pass

        plt.grid(True)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(loc="upper right")
        plt.savefig(os.path.join(self.log_dir, 'epoch_loss.png'))
        plt.cla()
        plt.close("all")

def show_config(**kwargs):
    print('Configurations')
    print('_' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('_' * 70)
    for k, v in kwargs.items():
        print('|%25s | %40s|' % (str(k), str(v)))
    print('_' * 70)


#将图像转化为RGB图像
def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert(('RGB'))
        return image


def preprocess_input(image):
    image = image / 255.0
    return image


# 设置DataLoader种子
def worker_init_fn(work_id, rank, seed):
    worker_seed = rank + seed
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


# 对图像进行resize
def resize_image(image, size):
    iw, ih = image.size
    w, h = size

    scale = min(w/iw, h/ih)
    nw = int(scale * iw)
    nh = int(scale * ih)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))

    return new_image, nw, nh


# 获得学习率
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

