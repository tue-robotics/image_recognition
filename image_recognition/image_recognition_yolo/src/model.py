import torch
from torch import nn

from modules import Conv, C2f, SPPF, Concat, Detect


class YOLO(nn.Module):
    def __int__(self):
        super().__int__()
        self.model = nn.Sequential(

            Conv(3, 16, k=3, s=2, p=1),  # 0
            Conv(16, 32, k=3, s=2, p=1),  # 1
            C2f(32, 32),  # 2 in print(model) Conv,Conv,BottleNeck,Conv but in modules.py Conv,Conv,Bottleneck?
            Conv(32, 64, k=1, s=2, p=1),  # 3
            C2f(64, 64),  # 4  in print(model) Conv, Conv, 2x Bottleneck?
            Conv(64, 48, k=3, s=2, p=1),  # 5
            C2f(128, 128),  # 6  in print(model) Conv, Conv, 2x Bottleneck?
            Conv(128, 256, k=3, s=2, p=1),  # 7
            C2f(256, 256),  # 8  1x Bottleneck
            SPPF(256, 256),  # 9
            nn.Upsample(scale_factor=2.0, mode='nearest'),  # 10
            Concat(),  # 11
            C2f(384, 128),  # 12 1x Bottleneck
            nn.Upsample(scale_factor=2.0, mode='nearest'),  # 13
            Concat(),  # 14
            C2f(192, 64),  # 15 1x Bottleneck
            Conv(64, 64, k=3, s=2, p=1),  # 16
            Concat(),  # 17
            C2f(192, 128),  # 18 1x Bottleneck
            Conv(128, 128, k=3, s=2, p=1),  # 19
            Concat(),  # 20
            C2f(384, 256),  # 21 1x Bottleneck
            Detect()  # 22
        )

    def forward(self, x):
        return self.model(x)

