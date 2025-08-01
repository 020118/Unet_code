import torch
import torch.nn as nn
import math
from typing import Union
from torch.hub import load_state_dict_from_url

class vggnet(nn.Module):
    def __init__(self, features, dropout=0.5, num_classes:int=1000):
        super(vggnet, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7,7))
        self.classifier = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(4096, num_classes)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        feat1 = self.features[:4](x)
        feat2 = self.features[4:9](feat1)
        feat3 = self.features[9:16](feat2)
        feat4 = self.features[16:23](feat3)
        feat5 = self.features[23:-1](feat4)
        return [feat1, feat2, feat3, feat4, feat5]

cfgs = {
    'D':[64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
}


def make_layer(cfgs, batch_norm=False, in_channels=3) -> nn.Sequential:
    layer = []
    for i in cfgs:
        if i == 'M':
            layer += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, i, kernel_size=3, padding=1)
            if batch_norm:
                layer += [conv2d, nn.BatchNorm2d(i), nn.ReLU(inplace=True)]
            else:
                layer += [conv2d, nn.ReLU(inplace=True)]
            in_channels = i
    return nn.Sequential(*layer)


def vgg16(pretrained, in_channels=3, **kwargs):
    features = make_layer(cfgs['D'], batch_norm=False, in_channels=in_channels)
    model = vggnet(features=features, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url("https://download.pytorch.org/models/vgg16-397923af.pth", model_dir="./model_data")
        model.load_state_dict(state_dict)

    del model.avgpool
    del model.classifier
    
    return model