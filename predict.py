import torch
import cv2
import time
import numpy as np
from PIL import Image
from unet import Unet, Unet_onnx


if __name__ == "__main__":
    mode = 'fps'
    count = False
    name_classes = ["background","aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

    video_path = 0
    video_save_pth = ''
    video_fps = 25.0

    test_interval = 100
    fps_image_path = 'img/street.jpg'

    dir_original_path = 'img/'
    dir_save_path = 'img_out/'

    simplify = True
    onnx_save_path = 'model_data/models.onnx'


    if mode != 'predict_onnx':
        unet = Unet()
    else:
        yolo = Unet_onnx()

    if mode == 'predict':
        while True:
            img = input("Input image filename:")
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = unet.detect_image(image, count=False, name_classes=name_classes)
                r_image.show()
    
    elif mode == 'video':
        capture = cv2.VideoCapture(video_path)
        if video_save_pth != '':
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out = cv2.VideoWriter(video_save_pth, fourcc, video_fps, size)
        
        ref, frame = capture.read()
        if not ref:
            raise ValueError("未能正确读取摄像头（视频），请注意是否正确安装摄像头（是否正确填写视频路径）。")
        
        fps = 0.0
        while(True):
            t1 = time.time()
            ref, frame = capture.read()
            if not ref:
                break

            # 格式转变  BGR->RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(np.uint8(frame))
            frame = np.array(unet.detect_image(frame))
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            fps = (fps + (1./(time.time() - t1))) / 2
            print(f'fps={fps}')
            frame = cv2.putText(frame, f'fps={fps}', (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('video', frame)
            c = cv2.waitKey(1) & 0xff
            if video_save_pth != '':
                out.write(frame)
            
            if c == 27:
                capture.release()
                break
        print('video detection done!')
        capture.release()
        if video_save_pth != '':
            print('save processes video to the path :' + video_save_pth)
            out.release()
        cv2.destroyAllWindows()

    elif mode == 'fps':
        img = Image.open('img/street.jpg')
        tact_time = unet.get_FPS(img, test_interval)
        print(str(tact_time) + ' seconds, ' + str(1./tact_time) + 'FPS, @batch_size 1')
    
    elif mode == 'dir_predict':
        import os
        from tqdm import tqdm

        img_names = os.listdir(dir_original_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_pth = os.path.join(dir_original_path, img_name)
                image = Image.open(image_pth)
                r_image = unet.detect_image(image)
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                r_image.save(os.path.join(dir_save_path, img_name))
    
    elif mode == 'export_onnx':
        unet.convert_to_onnx(simplify, onnx_save_path)
    
    elif mode == 'predict_onnx':
        while True:
            img = input('input image filename:')
            try:
                image = Image.open(img)
            except:
                print('open error! try again!')
            else:
                r_image = yolo.detect_image(image)
                r_image.show()
    else:
        raise AssertionError("please specify the correct mode: 'predict', 'video', 'fps', 'dir_predict'")








