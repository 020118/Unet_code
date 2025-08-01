import torch
from utils import cvtColor, resize_image, preprocess_input
import numpy as np
import torch.nn.functional as F 
import cv2
import os
import shutil
from PIL import Image
from tqdm import tqdm
from utils_metrics import compute_miou
from matplotlib import pyplot as plt


class EvalCallback():
    def __init__(self, net, input_shape, num_classes, image_ids, dataset_pth, log_dir,
                 cuda, miou_out_pth=".temp_miou_out", eval_flag=True, period=1):
        super(EvalCallback, self).__init__()
        self.input_shape = input_shape
        self.cuda = cuda
        self.net = net
        self.period = period
        self.eval_flag = eval_flag
        self.dataset_pth = dataset_pth
        self.miou_out_pth = miou_out_pth
        self.image_ids = image_ids
        self.num_classes = num_classes
        self.log_dir = log_dir

        self.image_ids = [image_id.split()[0] for image_id in image_ids]
        self.mious = [0]
        self.epoches = [0]
        if self.eval_flag:
            with open(os.path.join(self.log_dir, 'epoch_miou.txt'), 'a') as f:
                f.write(str(0))
                f.write('\n')

    def get_miou_png(self, image):
        #将图像转换成RGB图像
        image = cvtColor(image)
        original_h = np.array(image).shape[0]
        original_w = np.array(image).shape[1]

        image_data, nw, nh = resize_image(image, (self.input_shape[1], self.input_shape[0]))

        #添加上batch_size维度
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image, np.float32)), (2,0,1), 0))

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            
            # 图片传入进行预测
            pr = self.net(images)[0]

            # 取出每个像素点的种类
            pr = F.softmax(pr.permute(1,2,0), dim=-1).cpu().numpy()

            #将灰条部分截掉
            pr = pr[int((self.input_shape[0]-nh)//2) : int((self.input_shape[0] - nh)//2 + nh),
                    int((self.input_shape[1]-nw)//2) : int((self.input_shape[1]-nw)//2 + nw)]
            
            #将图片resize回原始尺寸
            pr = cv2.resize(pr, (original_w, original_h), interpolation=cv2.INTER_LINEAR)

            pr = pr.argmax(axis=-1)

        image = Image.fromarray(np.uint8(pr))
        return image
    
    def on_epoch_end(self, epoch, model_eval):
        if epoch % self.period == 0 and self.eval_flag:
            self.net = model_eval
            gt_dir = os.path.join(self.dataset_pth, 'VOC2007/SegmentationClass/')
            pred_dir = os.path.join(self.miou_out_pth, 'detection-results')
            if not os.path.exists(self.miou_out_pth):
                os.makedirs(self.miou_out_pth)
            if not os.path.exists(pred_dir):
                os.makedirs(pred_dir)
            print("Get miou.")

            for image_id in tqdm(self.image_ids):
                #从文件读取图像
                image_pth = os.path.join(self.dataset_pth, 'VOC2007/SegmentationClass' + image_id + '.jpg')
                image = Image.open(image_pth)

                #获得预测txt
                image = self.get_miou_png(image)
                image.save(os.path.join(pred_dir, image_id + '.png'))

            print("Caculate miou.")
            _, IoUs, _, _ = compute_miou(gt_dir, pred_dir, self.image_ids, self, None)
            temp_miou = np.nanmean(IoUs) * 100

            self.mious.append(temp_miou)
            self.epoches.append(epoch)

            with open(os.path.join(self.log_dir, 'epoch_miou.txt'), 'a') as f:
                f.write(str(temp_miou))
                f.write('\n')

            plt.figure()
            plt.plot(self.epoches, self.mious, 'red', linewidth=2, label='train miou')

            plt.grid(True)
            plt.xlabel('Epoch')
            plt.ylabel('Miou')
            plt.title('A Miou curve')
            plt.legend(loc='upper right')

            plt.savefig(os.path.join(self.log_dir, 'epoch_miou.png'))
            plt.cla()
            plt.close('all')

            print('Get miou done')
            shutil.rmtree(self.miou_out_pth)



