import torch
import os 
import csv
import numpy as np
from os.path import join
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt


def fast_hist(a, b, n):
    # a和b都是转化一维的数组
    # 生成掩码k，表示有效像素的位置
    k = (a >= 0) & (a < n)
    
    # 使用（n*a + b）来产生一个一维索引，用此来表示a和b在每个像素点的表示情况
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)


def per_class_iu(hist):
    return np.diag(hist) / np.maximum(hist.sum(1)+hist.sum(0)-np.diag(hist), 1)


def per_class_PA_recall(hist):
    return np.diag(hist) / np.maximum(hist.sum(1), 1)


def per_class_Precision(hist):
    return np.diag(hist) / np.maximum(hist.sum(0), 1)

def per_Accuracy(hist):
    return np.sum(np.diag(hist)) / np.maximum(np.sum(hist), 1) 

def compute_miou(gt_dir, pred_dir, png_name_list, num_classes, name_classes=None):
    print("Num classes", num_classes)

    hist = np.zeros((num_classes, num_classes))
    
    # 获得验证集真实标签路径列表以及验证集分割结果标签路径列表
    gt_imgs = [join(gt_dir, x + ".png") for x in png_name_list]
    pred_imgs = [join(pred_dir, x + ".png") for x in png_name_list]

    # 读取每一个图片-标签对
    for idx in range(len(gt_imgs)):
        pred = np.array(Image.open(pred_imgs[idx]))
        label = np.array(Image.open(gt_imgs[idx]))

        # 如果分割结果和真实标签的大小不同，则这张图片不计算
        if len(pred.flatten()) != len(label.flatten()):
            print("Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}".format(
                len(label.flatten()), len(pred.flatten()), gt_imgs[idx], pred_imgs[idx]))
            continue

        # 对一张图片计算21*21的hist矩阵，并累加
        hist += fast_hist(label.flatten(), pred.flatten(), num_classes) 

        # 每计算10张图片就计算当前完成计算的图片的平均miou值
        if name_classes is not None and idx > 0 and idx % 10 == 0:
            print("{:d} / {:d}: miou-{:0.2f}%, mPA-{:0.2f}%, Accuracy-{:0.2f}%".format(
                idx, len(gt_imgs), 100 * np.nanmean(per_class_iu(hist)),
                100 * np.nanmean(per_class_PA_recall(hist)), 100 * per_Accuracy(hist)
            ))
    
    # 计算验证集所有图片的逐类别miou的值
    IoUs = per_class_iu(hist)
    PA_recall = per_class_PA_recall(hist)
    Precision = per_class_Precision(hist)

    # 逐类别输出miou的值
    if name_classes is not None:
        for idx_class in range(num_classes):
            print("===>" + name_classes[idx_class] + ':\tIou-' + str(round(IoUs[idx_class]*100, 2))
                  + ';  Recall (equal to the PA)-' + str(round(PA_recall[idx_class]*100, 2))
                  + '; Precision-' + str(round(Precision[idx_class]*100, 2)))
    
    # 在所有验证集图片上计算所有类别的平均miou的值, 计算时忽略NaN值
    print('===> mIoU: ' + str(round(np.nanmean(IoUs)*100, 2))
          + '; mPA: ' + str(round(np.nanmean(PA_recall)*100, 2))
          + '; Precision: ' + str(round(np.nanmean(Precision)*100, 2)))
    return np.array(hist, dtype=int), IoUs, PA_recall, Precision


def f_score(inputs, targets, beta=1, smooth=1e-5, threshold=0.5):
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = targets.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c),-1)
    temp_target = targets.view(n, -1, ct)

    temp_inputs = torch.gt(temp_inputs, threshold).float()  # 二值化，大于threshold的设为1，小于的设为0
    tp = torch.sum(temp_target[...,:-1] * temp_inputs, axis=[0,1])
    fp = torch.sum(temp_inputs                       , axis=[0,1]) - tp
    fn = torch.sum(temp_target[...,:-1]              , axis=[0,1]) - tp

    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    score = torch.mean(score)
    return score


def adjust_axes(r, t, fig, axes):
    bb = t.get_window_extent(renderer=r)
    text_width_inches = bb.width / fig.dpi
    current_fig_width = fig.get_figwidth()
    new_fig_width = current_fig_width + text_width_inches
    propotion = new_fig_width / current_fig_width
    x_lim = axes.get_xlim()
    axes.set_xlim([x_lim[0], x_lim[1] * propotion])


def draw_plot_func(values, name_classes, plot_title, x_label, output_path, tick_font_size=12, plt_show=True):
    fig = plt.gcf()
    axes = plt.gca()
    plt.barh(range(len(values)), values, color='royalblue')
    plt.title(plot_title, fontsize=tick_font_size+2)
    plt.xlabel(x_label, fontsize=tick_font_size)
    plt.yticks(range(len(values)), name_classes, fontsize=tick_font_size)
    r = fig.canvas.get_renderer()
    for i, value in enumerate(values):
        str_val = " " + str(value)
        if value < 1.0:
            str_val = " {0:.2f}".format(value)
        t = plt.text(value, i, str_val, color='royalblue', va='center', fontweight='bold')
        if i == (len(values) - 1):
            adjust_axes(r, t, fig, axes)
    
    plt.tight_layout()
    plt.savefig(output_path)
    if plt_show:
        plt.show()
    plt.close()


def show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes, tick_font_size=12):
    draw_plot_func(IoUs, name_classes, 'mIoU = {0:.2f}%'.format(np.nanmean(IoUs)*100), 'Intersection over Union',
                   os.path.join(miou_out_path, 'mIoUs.png'), tick_font_size=tick_font_size, plt_show=True)
    print('Save mIoU out to ' + os.path.join(miou_out_path, 'mIoUs.png'))

    draw_plot_func(PA_Recall, name_classes, "mPA = {0:.2f}%".format(np.nanmean(PA_Recall)*100), "Pixel Accuracy", \
        os.path.join(miou_out_path, "mPA.png"), tick_font_size = tick_font_size, plt_show = False)
    print("Save mPA out to " + os.path.join(miou_out_path, "mPA.png"))
    
    draw_plot_func(PA_Recall, name_classes, "mRecall = {0:.2f}%".format(np.nanmean(PA_Recall)*100), "Recall", \
        os.path.join(miou_out_path, "Recall.png"), tick_font_size = tick_font_size, plt_show = False)
    print("Save Recall out to " + os.path.join(miou_out_path, "Recall.png"))

    draw_plot_func(Precision, name_classes, "mPrecision = {0:.2f}%".format(np.nanmean(Precision)*100), "Precision", \
        os.path.join(miou_out_path, "Precision.png"), tick_font_size = tick_font_size, plt_show = False)
    print("Save Precision out to " + os.path.join(miou_out_path, "Precision.png"))

    with open(os.path.join(miou_out_path, 'confusion_matirx.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer_list = []
        writer_list.append([' '] + [str(i) for i in name_classes])
        for i in range(len(hist)):
            writer_list.append([name_classes[i]] + [str(x) for x in hist[i]])
        writer.writerows(writer_list)
    print('Save confusion matrix to ' + os.path.join(miou_out_path, 'confusion_matrix.csv'))




