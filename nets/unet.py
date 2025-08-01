import torch
import torch.nn as nn
from nets.resnet import resnet50
from nets.vgg import vgg16


class sampleup(nn.Module):
    def __init__(self, in_size, out_size):
        super(sampleup, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        self.upconv = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, input1, input2):
        output = torch.cat([input1, self.upconv(input2)], dim=1)
        output = self.conv1(output)
        output = self.relu(output)
        output = self.conv2(output)
        output = self.relu(output)
        return output
        

class unet(nn.Module):
    def __init__(self, num_classes=21, pretrained=False, backbone='vgg'):
        super(unet, self).__init__()
        if backbone == 'vgg':
            self.vgg = vgg16(pretrained)
            in_filters = [192, 384, 768, 1024]
        elif backbone == 'resnet50':
            self.resnet50 = resnet50(pretrained)
            in_filters = [192, 512, 1024, 3072]
        else:
            raise ValueError("unsupported backbone - {}, use vgg, resnet50".format(backbone))
        out_filter = [64, 128, 256, 512]

        self.up_concat4 = sampleup(in_filters[3], out_filter[3])
        self.up_concat3 = sampleup(in_filters[2], out_filter[2])
        self.up_concat2 = sampleup(in_filters[1], out_filter[1])
        self.up_concat1 = sampleup(in_filters[0], out_filter[0])

        if backbone == 'resnet50':
            self.up_conv = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(out_filter[0], out_filter[0], kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(out_filter[0], out_filter[0], kernel_size=3, padding=1),
                nn.ReLU()
            )
        else:
            self.up_conv = None

        self.final = nn.Conv2d(out_filter[0], num_classes, kernel_size=1)
        self.backbone = backbone

    def forward(self, x):
        if self.backbone == 'vgg':
            [feat1, feat2, feat3, feat4, feat5] = self.vgg.forward(x)
        elif self.backbone == 'resnet50':
            [feat1, feat2, feat3, feat4, feat5] = self.resnet50.forward(x)
        
        up4 = self.up_concat4(feat4, feat5)
        up3 = self.up_concat3(feat3, up4)
        up2 = self.up_concat2(feat2, up3)
        up1 = self.up_concat1(feat1, up2)
        
        if self.up_conv is not None:
            up1 = self.up_conv(up1)
        output = self.final(up1)

        return output
    
    def freeze_backbone(self):
        if self.backbone == 'vgg':
            for param in self.vgg.parameters():
                param.requires_grad = False
        elif self.backbone == 'resnet50':
            for param in self.resnet50.parameters():
                param.requires_grad = False
    
    def unfreeze_backbone(self):
        if self.backbone == 'vgg':
            for param in self.vgg.parameters():
                param.requires_grad = True
        elif self.backbone == 'resnet50':
            for param in self.resnet50.parameters():
                param.requires_grad = True         
            

            