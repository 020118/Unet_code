import torch
import os 
import cv2
import numpy as np
from torch.utils.data.dataset import Dataset
from PIL import Image
from utils import cvtColor
from utils import preprocess_input

class UnetDataset(Dataset):
    def __init__(self, annolation_lines, input_shape, num_classes, train, dataset_pth):
        super(UnetDataset, self).__init__()
        self.annolation_lines = annolation_lines
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.train = train
        self.dataset_pth = dataset_pth
    
    def __len__(self):
        return len(self.annolation_lines)

    def __getitem__(self, index):
        annolation_line = self.annolation_lines[index]
        name = annolation_line.split()[0]

        #从文件读取图像
        jpg = Image.open(os.path.join(os.path.join(self.dataset_pth, "VOC2007/JPEGImages"), name + '.jpg'))
        png = Image.open(os.path.join(os.path.join(self.dataset_pth, "VOC2007/SegmentationClass"), name + '.png'))
        #数据增强
        jpg, png = self.get_random_data(jpg, png, self.input_shape, random=self.train)

        jpg = np.transpose(preprocess_input(np.array(jpg, np.float64)), [2,0,1])
        png = np.array(png)
        png[png >= self.num_classes] = self.num_classes

        seg_labels = np.eye(self.num_classes+1)[png.reshape([-1])]
        seg_labels = seg_labels.reshape((int(self.input_shape[0]), int(self.input_shape[1]), self.num_classes+1))

        return jpg, png, seg_labels


    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, image, label, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.3, random=True):
        image = cvtColor(image)
        label = Image.fromarray(np.array(label))
        
        #获得图像和目标的高宽
        iw, ih = image.size  #PIL的image.size返回的是宽*高
        h, w = input_shape   #这里返回的是高*宽

        if not random:
            iw, ih = image.size
            scale = min(w/iw, h/ih)
            nw = int(scale * iw)
            nh = int(scale * ih)

            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', [w, h], (128, 128, 128))
            new_image.paste(image, ((w-nw)//2, (h-nh)//2))

            label = label.resize((nw, nh), Image.NEAREST)
            new_label = Image.new('L', [w, h], (128, 128, 128))
            new_label.paste(label, ((w-nw)//2, (h-nh)//2))
            return new_image, new_label
        
        #对图像进行缩放并进行长和宽的扭曲
        new_ar = iw/ih * self.rand(1-jitter, 1+jitter) / self.rand(1-jitter, 1+jitter)
        scale = self.rand(0.25, 2)
        if new_ar < 1:
            nh = int(h * scale)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)
        label = label.resize((nw, nh), Image.NEAREST)

        #翻转图像
        flip = self.rand() < .5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)

        #将图像多余部分加上灰条
        dx = int(self.rand(0, w-nw))
        dy = int(self.rand(0, h-nh))
        new_image = Image.new('RGB', [w, h], (128, 128, 128))
        new_label = Image.new('L', [w, h], (0))
        new_image.paste(image, (dx, dy))
        new_label.paste(label, (dx, dy))
        image = new_image
        label = new_label
        image_data = np.array(image, np.uint8)

        #进行色域变换
        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1

        hue, sat, val = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype = image_data.dtype

        #应用变换（设立查找表提高图像变换的效率）
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x*r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x*r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x*r[2], 0, 255).astype(dtype)

        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)

        return image_data, label
    

# DataLoader中用在collate_fn中
def unet_dataset_collate(batch):
    images = []
    pngs = []
    seg_labels = []
    for img, png, labels in batch:
        images.append(img)
        pngs.append(png)
        seg_labels.append(labels)
    images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    pngs = torch.from_numpy(np.array(pngs)).long()
    seg_labels = torch.from_numpy(np.array(seg_labels)).type(torch.FloatTensor)
    return images, pngs, seg_labels


