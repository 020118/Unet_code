import torch
import colorsys
import copy
import cv2
import time
from nets.unet import unet
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from PIL import Image
from utils import show_config, cvtColor, resize_image, preprocess_input


class Unet(object):
    _defaults = {
        'model_path': 'model_data/unet_vgg_voc.pth',
        'num_classes': 21,
        'backbone': 'vgg',
        'input_size': [512, 512],
        'mix_type': 0,
        'cuda': True
    }
    
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
        
        if self.num_classes <= 21:
            self.colors =  [(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128), 
                            (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128), 
                            (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128), 
                            (128, 64, 12)]
        else:
            hsv_tuples  = [(x/self.num_classes, 1., 1.) for x in range(self.num_classes)]
            self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            self.colors = list(map(lambda x: (int(x[0]*255), int(x[1]*255), int(x[2]*255)), self.colors))

        self.generate()
        show_config(**self._defaults)
  

    def generate(self, onnx=False):
        # 加载模型和模型权重，设置推理模式
        self.net = unet(num_classes=21, backbone=self.backbone)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net = self.net.eval()
        print(f'{self.model_path} model, and classes loaded.')
        if not onnx:
            if self.cuda:
                self.net = nn.DataParallel(self.net)
                self.net = self.net.cuda()
    

    def detect_image(self, image, count=False, name_classes=None):
        image = cvtColor(image)

        # 对输入图像进行备份，后面用作绘图
        old_img = copy.deepcopy(image)
        original_h = np.array(old_img).shape[0]
        original_w = np.array(old_img).shape[1]

        # 给图像增加灰条
        image_data, nw, nh = resize_image(image, (self.input_size[1], self.input_size[0]))

        # 加上batch size维度
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2,0,1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()

            pr = self.net(images)[0]
            pr = F.softmax(pr.permute(1,2,0), dim=-1).cpu().numpy()

            # 将灰条部分截掉
            pr = pr[int((self.input_size[0]-nh)//2) : int((self.input_size[0]-nh)//2 + nh),
                    int((self.input_size[1]-nw)//2) : int((self.input_size[1]-nw)//2 + nw)]
            
            # 对图片resize
            pr = cv2.resize(pr, (original_w, original_h), interpolation = cv2.INTER_LINEAR)

            # 取出每个像素点的种类
            pr = pr.argmax(axis=-1)

        if count:
            classes_nums = np.zeros([self.num_classes])
            total_points_num = original_h * original_w
            print('-' * 63)
            print("|%25s | %15s | %15s|" % ('key', 'value', 'ratio'))
            print('-' * 63)
            for i in range(self.num_classes):
                num = np.sum(i == pr)
                ratio = num / total_points_num * 100
                if num > 0:
                    print('|%25s | %15s| %14.2f%%|' % (str(name_classes[i]), str(num), ratio))
                    print('-' * 63)
                classes_nums[i] = num
            print('classes_nums: ', classes_nums)

        if self.mix_type == 0:
            seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])], [original_h, original_w, -1])
            image = Image.fromarray(np.uint8(seg_img))
            image = Image.blend(old_img, image, 0.7)
        
        elif self.mix_type == 1:
            seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])], [original_h, original_w, -1])
            image = Image.fromarray(np.uint8(seg_img))

        elif self.mix_type == 2:
            seg_img = (np.expand_dims(pr != 0, -1) * np.array(old_img, np.float32)).astype('uint8')
            image = Image.fromarray(np.uint8(seg_img))

        return image

    
    def get_FPS(self, image, test_interval):
        image = cvtColor(image)
        image_data, nw, nh = resize_image(image, (self.input_size[1], self.input_size[0]))
        image_data = np.expand_dims(preprocess_input(np.transpose(np.array(image_data, np.float32), (2, 0, 1))), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()

            pr = self.net(images)[0]
            pr = F.softmax(pr.permute(1, 2, 0), -1).cpu().numpy().argmax(axis=-1)
            pr = pr[int((self.input_size[0] - nh) // 2) : int((self.input_size[0] - nh) // 2 + nh),
                    int((self.input_size[1] - nw) // 2) : int((self.input_size[1] - nw) // 2 + nw)]
            
        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
                pr = self.net(images)[0]
                pr = F.softmax(pr.permute(1, 2, 0), -1).cpu().numpy().argmax(axis=-1)
                pr = pr[int((self.input_size[0] - nh) // 2) : int((self.input_size[0] - nh) // 2 + nh),
                        int((self.input_size[1] - nw) // 2) : int((self.input_size[1] - nw) // 2 + nw)]
        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time
    
    def convert_to_onnx(self, simplify, model_path):
        import onnx
        self.generate(onnx=True)

        im = torch.zeros(1, 3, *self.input_shape).to('cpu')
        input_layer_names = ['images']
        output_layer_names = ['output']

        # 导出模型
        print(f'start export with onnx {onnx.__version__}.')
        torch.onnx.export(self.net,
                          im,
                          f=model_path,
                          verbose=False,
                          opset_version=12,
                          training=torch.onnx.TrainingMode.EVAL,
                          do_constant_folding=True,
                          input_names=input_layer_names,
                          output_names=output_layer_names,
                          dynamic_axes=None)

        # check
        model_onnx = onnx.load(model_path)
        onnx.checker.check_model(model_onnx)

        if simplify:
            import onnxsim
            print(f'simplify with onnxsim: {onnxsim.__version__}.')
            model_onnx, check = onnxsim.simplify(
                model_onnx,
                dynamic_input_shape=False,
                input_shapes=None
            )
            assert check, 'assert check failed'
            onnx.save(model_onnx, model_path)

        print('onnx model save as {}'.format(model_path))

    def get_miou_png(self, image):
        image = cvtColor(image)
        original_h = np.array(image).shape[0]
        original_w = np.array(image).shape[1]

        image_data, nw, nh = resize_image(image, (self.input_size[1], self.input_size[0]))
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()

            pr = self.net(images)[0]
            pr = F.softmax(pr.permute(1,2,0), dim=-1).cpu().numpy()

            pr = pr[int((self.input_size[0] - nh) // 2) : int((self.input_size[0] - nh) // 2 + nh), 
                    int((self.input_size[1] - nw) // 2) : int((self.input_size[1] - nw) // 2 + nw)]
            pr = cv2.resize(pr, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
            pr = pr.argmax(axis=-1)
        
        image = Image.fromarray(np.uint8(pr))
        return image


class Unet_onnx(object):
    _defaults = {
        'onnx_path': 'model_data/models.onnx',
        'num_classes': 21,
        'backbone': 'vgg',
        'input_size': [512, 512],
        'mix_type': 0
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._default:
            return cls._default[n]
        else:
            return "unrecognized attribute name: '" + n + "'"
        
    def __init__(self, **kwargs):
        self.__dict__.update(self._default)
        for name, value in kwargs.items():
            setattr(self, name, value)
            self._default[name] = value
        
        import onnxruntime
        self.onnx_session = onnxruntime.InferenceSession(self.onnx_path)
        self.input_name = self.get_input_name()
        self.output_name = self.get_output_name()

        if self.num_classes <= 21:
            self.colors =  [(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128), 
                            (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128), 
                            (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128), 
                            (128, 64, 12)]
        else:
            hsv_tuple = [(x/self.num_classes, 1., 1.) for x in range(self.num_classes)]
            self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuple))
            self.colors = list(map(lambda x: (int(x[0]*255), int(x[1]*255), int(x[2]*255)), self.colors))
        
        show_config(**self._defaults)

    def get_input_name(self):
        # 获得所有的输入node
        input_name = []
        for node in self.onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name
    
    def get_output_name(self):
        output_name = []
        for node in self.onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name
        
    def detect_image(self, image, count=False, name_classes=None):
        image = cvtColor(image)

        old_img = copy.deepcopy(image)
        original_h = np.array(image).shape[0]
        original_w = np.array(image).shape[1]

        image_data, nw, nh = resize_image(image, (self.input_size[1], self.input_size[0]))
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        input_feed = self.get_input_feed(image_data)
        pr = self.onnx_session.run(output_names=self.output_name, input_feed=input_feed)[0][0]

        def softmax(x, axis):
            x -= np.max(x, axis=axis, keepdims=True)
            f_x = np.exp(x) / np.sum(np.exp(x), axis=axis, keepdims=True)
            return f_x
        print(np.shape(pr))

        pr = softmax(np.transpose(pr, (1, 2, 0)), axis=-1)
        pr = pr[int((self.input_size[0] - nh) // 2) : int((self.input_size[0] - nh) // 2 + nh),
                int((self.input_size[1] - nw) // 2) : int((self.input_size[1] - nw) // 2 + nw)]
        pr = cv2.resize(pr, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
        pr = pr.argmax(axis=-1)

        if count:
            classes_nums = np.zeros([self.num_classes])
            total_points_num = original_w * original_h
            print('-' * 63)
            print("|%25s | %15s | %15s|"%("Key", "Value", "Ratio"))
            print('-' * 63)
            for i in range(self.num_classes):
                num = np.sum(pr == i)
                ratio = num / total_points_num * 100
                if num > 0:
                    print("|%25s | %15s | %14.2f%%|"%(str(name_classes[i]), str(num), ratio))
                    print('-' * 63)
                classes_nums[i] = num
            print("classes nums:", classes_nums)

        if self.mix_type == 0:
            seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])], [original_h, original_w, -1])
            image = Image.fromarray(np.uint8(seg_img))
            image = Image.blend(old_img, image, 0.7)
        
        elif self.mix_type == 1:
            seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])], [original_h, original_w, -1])
            image = Image.fromarray(np.uint8(seg_img))

        elif self.mix_type == 2:
            seg_img = (np.expand_dims(pr != 0, -1) * np.array(old_img, np.float32)).astype('uint8')
            image = Image.fromarray(np.uint8(seg_img))

        return image

    def get_input_feed(self, image_data):
        input_feed = {}
        for name in self.input_name:
            input_feed[name] = image_data
        return input_feed




            

