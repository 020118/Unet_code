import os
import torch
from unet import Unet
from tqdm import tqdm
from PIL import Image
from utils_metrics import compute_miou, show_results


if __name__ == '__main__':
    miou_mode = 0
    num_classes = 21
    name_classes = ["background","aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    VOCdevkit_path = 'VOCdevkit'

    image_ids = open(os.path.join(VOCdevkit_path, 'VOC2007/ImageSets/Segmentation/val.txt'), 'r').read().splitlines()
    gt_dir = os.path.join(VOCdevkit_path, "VOC2007/SegmentationClass/")
    miou_out_path = 'miou_out'
    pred_dir = os.path.join(miou_out_path, 'detection-results')

    if miou_mode == 0 or miou_mode == 1:
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)

        print('load model.')
        unet = Unet()
        print('load model done.')

        print('get predict results.')
        for image_id in tqdm(image_ids):
            image_path = os.path.join(VOCdevkit_path, 'VOC2007/JPEGImages/' + image_id + '.jpg')
            image = Image.open(image_path)
            image = unet.get_miou_png(image)
            image.save(os.path.join(pred_dir, image_id + '.png'))
        print('get predict results done.')

    if miou_mode == 0 or miou_mode == 2:
        print('get miou.')
        hist, IoUs, PA_Recall, Precision = compute_miou(gt_dir, pred_dir, image_ids, num_classes, name_classes)
        print('get miou done.')
        show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes)
        

        


