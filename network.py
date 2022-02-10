import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class ResNet(paddle.nn.Layer):
    def __init__(self):
        super(ResNet, self).__init__()
        r50 = paddle.vision.models.resnet50(True)
        self.cv1 = r50.conv1
        self.bn1 = r50.bn1
        self.maxp = r50.maxpool
        self.relu = r50.relu
        self.layer1 = r50.layer1
        self.layer2 = r50.layer2
        self.layer3 = r50.layer3
        self.layer4 = r50.layer4

    def forward(self, x):
        x0 = self.relu(self.bn1(self.cv1(x)))
        x1 = self.maxp(x0)
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        return x0, x2, x3, x4, x5

    def init_weight(self):
        print("bkbone finished!")


class MappingModule(nn.Layer):
    def __init__(self):
        super(MappingModule, self).__init__()
        self.cv1 = nn.Sequential(
            nn.Conv2D(256, 64, 1),
            nn.BatchNorm2D(64),
            nn.ReLU()
        )
        self.cv2 = nn.Sequential(
            nn.Conv2D(512, 64, 1),
            nn.BatchNorm2D(64),
            nn.ReLU()
        )
        self.cv3 = nn.Sequential(
            nn.Conv2D(1024, 64, 1),
            nn.BatchNorm2D(64),
            nn.ReLU()
        )
        self.cv4 = nn.Sequential(
            nn.Conv2D(2048, 64, 1),
            nn.BatchNorm2D(64),
            nn.ReLU()
        )

    def forward(self, out1, out2, out3, out4, out5):
        out2 = self.cv1(out2)
        out3 = self.cv2(out3)
        out4 = self.cv3(out4)
        out5 = self.cv4(out5)
        return out1, out2, out3, out4, out5


class SA(nn.Layer):
    def __init__(self):
        super(SA, self).__init__()
        self.cv1 = nn.Sequential(
            nn.Conv2D(2, 1, 7, 1, 3),
            nn.Sigmoid()
        )

    def forward(self, x):
        channel_mean = paddle.mean(x, axis=1, keepdim=True)
        channel_maxs = paddle.max(x, axis=1, keepdim=True)
        weight_mean = paddle.concat([channel_mean, channel_maxs], 1)
        weight_mean = self.cv1(weight_mean)
        x = x * weight_mean
        return x


class CA(nn.Layer):
    def __init__(self, in_ch, reduction=64):
        super(CA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.max_pool = nn.AdaptiveMaxPool2D(1)
        self.fc1 = nn.Conv2D(in_ch, in_ch // reduction, 1, bias_attr=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2D(in_ch // reduction, in_ch, 1, bias_attr=False)

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out_cha = avg_out + max_out
        out_cha = F.sigmoid(out_cha)
        x = x * out_cha
        return x


class DOM(nn.Layer):
    def __init__(self):
        super(DOM, self).__init__()
        self.refine1 = nn.Sequential(
            nn.Conv2D(64, 64, (7, 7), 1, (3, 3)),
            nn.BatchNorm2D(64),
            nn.ReLU(),
            nn.Conv2D(64, 64, (7, 7), 1, (3, 3)),
            nn.BatchNorm2D(64),
        )
        self.refine2 = nn.Sequential(
            nn.Conv2D(64, 64, (5, 5), 1, (2, 2)),
            nn.BatchNorm2D(64),
            nn.ReLU(),
            nn.Conv2D(64, 64, (5, 5), 1, (2, 2)),
            nn.BatchNorm2D(64),
        )
        self.refine3 = nn.Sequential(
            nn.Conv2D(64, 64, (3, 3), 1, (1, 1)),
            nn.BatchNorm2D(64),
            nn.ReLU(),
            nn.Conv2D(64, 64, (3, 3), 1, (1, 1)),
            nn.BatchNorm2D(64),
        )

    def forward(self, x):
        x1 = F.relu(self.refine1(x) + x)
        x2 = F.relu(self.refine2(x1) + x1)
        x3 = F.relu(self.refine3(x2) + x2)
        return x3


class PFEM(nn.Layer):
    def __init__(self):
        super(PFEM, self).__init__()
        self.cv1 = nn.Sequential(
            nn.Conv2D(64, 64, 3, 1, 7, dilation=7),
            nn.BatchNorm2D(64),
            nn.ReLU(),
            nn.Conv2D(64, 64, 3, 1, 1),
            nn.BatchNorm2D(64),
            nn.ReLU()
        )
        self.cv2 = nn.Sequential(
            nn.Conv2D(64 * 2, 64 * 2, 3, 1, 7, dilation=7),
            nn.BatchNorm2D(64 * 2),
            nn.ReLU(),
            nn.Conv2D(64 * 2, 64, 3, 1, 1),
            nn.BatchNorm2D(64),
            nn.ReLU()
        )
        self.cv3 = nn.Sequential(
            nn.Conv2D(64 * 3, 64 * 3, 3, 1, 7, dilation=7),
            nn.BatchNorm2D(64 * 3),
            nn.ReLU(),
            nn.Conv2D(64 * 3, 64, 3, 1, 1),
            nn.BatchNorm2D(64),
            nn.ReLU()
        )
        self.cv4 = nn.Sequential(
            nn.Conv2D(64 * 4, 64 * 4, 3, 1, 7, dilation=7),
            nn.BatchNorm2D(64 * 4),
            nn.ReLU(),
            nn.Conv2D(64 * 4, 64, 3, 1, 1),
            nn.BatchNorm2D(64),
            nn.ReLU()
        )
        self.cv5 = nn.Sequential(
            nn.Conv2D(64 * 5, 64 * 5, 3, 1, 7, dilation=7),
            nn.BatchNorm2D(64 * 5),
            nn.ReLU(),
            nn.Conv2D(64 * 5, 64, 3, 1, 1),
            nn.BatchNorm2D(64),
            nn.ReLU()
        )

    def forward(self, xx, y):
        x1 = F.interpolate(self.cv1(xx), size=y.shape[2:], mode='bilinear')
        x2 = F.interpolate(self.cv2(paddle.concat([x1, xx], 1)), size=y.shape[2:], mode='bilinear')
        x3 = F.interpolate(self.cv3(paddle.concat([x2, x1, xx], 1)), size=y.shape[2:], mode='bilinear')
        x4 = F.interpolate(self.cv4(paddle.concat([x3, x2, x1, xx], 1)), size=y.shape[2:], mode='bilinear')
        x5 = F.interpolate(self.cv5(paddle.concat([x4, x3, x2, x1, xx], 1)), size=y.shape[2:], mode='bilinear')
        xs = paddle.concat([x1, x2, x3, x4, x5, y], 1)
        return xs


class Fusion(nn.Layer):  # not write
    def __init__(self):
        super(Fusion, self).__init__()

    def forward(self, l, h):
        # fusion l and h
        return l


class DOMs(nn.Layer):  # low-level features
    def __init__(self):
        super(DOMs, self).__init__()
        self.lf1, self.lf2 = DOM(), DOM()
        self.cv = nn.Sequential(
            nn.Conv2D(64 * 2, 64, 3, 1, 1),
            nn.BatchNorm2D(64),
            nn.ReLU()
        )
        self.sa = SA()

    def forward(self, x1, x2):
        x1 = self.lf1(x1)
        x2 = self.lf2(x2)
        x2 = F.interpolate(x2, size=x1.shape[2:], mode='bilinear')
        xs = paddle.concat([x1, x2], 1)
        xs = self.cv(xs)
        xs = self.sa(xs)
        return xs


class PFEMs(nn.Layer):  # high-level features
    def __init__(self):
        super(PFEMs, self).__init__()
        self.fa4, self.fa5 = PFEM(), PFEM()
        self.ca = CA(64 * 2 * 6)
        self.cv = nn.Sequential(
            nn.Conv2D(64 * 2 * 6, 64, 1),
            nn.BatchNorm2D(64),
            nn.ReLU()
        )

    def forward(self, x4, x5):
        out4, out5 = self.fa4(x4, x4), self.fa5(x5, x5)
        out5 = F.interpolate(out5, size=x4.shape[2:], mode='bilinear')
        hf = paddle.concat([out4, out5], 1)
        hf = self.ca(hf)
        hf = self.cv(hf)
        return hf


class P4Net(nn.Layer):
    def __init__(self, cfg):
        super(P4Net, self).__init__()
        self.cfg = cfg
        self.bkbone = ResNet()
        self.mp = MappingModule()
        self.ld = DOMs()
        self.hd = PFEMs()
        self.fu = Fusion()
        for p in self.bkbone.parameters():
            p.optimize_attr['learning_rate'] /= 10.0
        self.linear = nn.Conv2D(64, 1, 3, 1, 1)

    def forward(self, x):
        x1, x2, x3, x4, x5 = self.bkbone(x)
        x1, x2, x3, x4, x5 = self.mp(x1, x2, x3, x4, x5)
        xl = self.ld(x2, x3)
        xh = self.hd(x4, x5)
        out = self.fu(xl, xh)
        out = F.interpolate(self.linear(out), mode='bilinear', size=x.shape[2:])
        return out


if __name__ == '__main__':
    import pandas as pd

    cag = pd.Series({'snapshot': False})
    f4 = P4Net(cag)
    x = paddle.rand((1, 3, 640, 640))
    y = f4(x)
    print(y[0].shape)
    total_params = sum(p.numel() for p in f4.parameters())
    print('total params : ', total_params)
    FLOPs = paddle.flops(f4, [1, 3, 352, 352],
                         print_detail=False)
    print(FLOPs)
