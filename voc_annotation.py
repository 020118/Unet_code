import torch
import random
import os
import numpy as np
from PIL import Image


# 对数据集进行划分，划分为训练集和验证集，并检查标签图像格式是否符合要求
trainval_percent = 1
train_percent = 0.9
VOCdevkit_path = 'VOCdevkit'

if __name__ == '__main__':
    random.seed(0)
    print("Generate txt in Imagesets")
    seg_file_path = os.path.join(VOCdevkit_path, 'VOC2007/SegmentationClass')
    save_base_path = os.path.join(VOCdevkit_path, 'VOC2007/ImageSets/Segmentation')

    temp_seg = os.listdir(seg_file_path)
    total_seg = []
    for seg in temp_seg:
        if seg.endswith('.png'):
            total_seg.append(seg)

    num = len(total_seg)
    list = range(num)
    tv = int(trainval_percent * num)
    tr = int(tv * train_percent)
    trainval = random.sample(list, tv)
    train = random.sample(trainval, tr)

    print('train and val size', tv)
    print('train size', tr)
    ftrainval = open(os.path.join(save_base_path, 'trainval.txt'), 'w')
    ftest = open(os.path.join(save_base_path, 'test.txt'), 'w')
    ftrain = open(os.path.join(save_base_path, 'train.txt'), 'w')
    fval = open(os.path.join(save_base_path, 'val.txt'), 'w')

    for i in list:
        name = total_seg[i][:-4] + '\n'
        if i in trainval:
            ftrainval.write(name)
            if i in train:
                ftrain.write(name)
            else:
                fval.write(name)
        else:
            ftest.write(name)

    ftrainval.close()
    ftrain.close()
    ftest.close()
    fval.close()
    print('Generate txt in Imagesets done.')

    print('check dataset format, this may take a while.')
    classes_num = np.zeros([256], dtype=int)
    for i in list:
        name = total_seg[i]
        png_file_name = os.path.join(seg_file_path, name)
        if not os.path.exists(png_file_name):
            raise ValueError("未检测到标签图片%s，请检查" % (png_file_name))
        
        png = np.array(Image.open(png_file_name), np.uint8)
        if len(np.shape(png)) > 2:
            print("标签图片%s的shape为%s，不属于灰度图或者八位彩图，请仔细检查数据集格式。"%(name, str(np.shape(png))))

        classes_num += np.bincount(np.reshape(png, [-1]), minlength=256)

    print("打印像素点的值与数量。")
    print('-' * 37)
    print("| %15s | %15s |"%("Key", "Value"))
    print('-' * 37)
    for i in range(256):
        if classes_num[i] > 0:
            print("| %15s | %15s |"%(str(i), str(classes_num[i])))
            print('-' * 37)

    if classes_num[255] > 0 and classes_num[0] > 0 and np.sum(classes_num[1:255]) == 0:
        print("检测到标签中像素点的值仅包含0与255，数据格式有误。")
    elif classes_num[0] > 0 and np.sum(classes_num[1:]) == 0:
        print("检测到标签中仅仅包含背景像素点，数据格式有误，请仔细检查数据集格式。")

    print("JPEGImages中的图片应当为.jpg文件、SegmentationClass中的图片应当为.png文件。")
    print("如果格式有误，参考:")
    print("https://github.com/bubbliiiing/segmentation-format-fix")

