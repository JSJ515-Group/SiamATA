from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F

from siamese.core.xcorr import xcorr_fast, xcorr_depthwise, xcorr_pixelwise
from siamese.models.head.pw_head import PWHead
from siamese.attention.fdta import FDTA


class BAN(nn.Module):
    def __init__(self):
        super(BAN, self).__init__()

    def forward(self, z_f, x_f):
        raise NotImplementedError


class UPChannelBAN(BAN):
    def __init__(self, feature_in=256, cls_out_channels=2):
        super(UPChannelBAN, self).__init__()

        cls_output = cls_out_channels
        loc_output = 4

        self.template_cls_conv = nn.Conv2d(feature_in,
                                           feature_in * cls_output, kernel_size=3)
        self.template_loc_conv = nn.Conv2d(feature_in,
                                           feature_in * loc_output, kernel_size=3)

        self.search_cls_conv = nn.Conv2d(feature_in,
                                         feature_in, kernel_size=3)
        self.search_loc_conv = nn.Conv2d(feature_in,
                                         feature_in, kernel_size=3)

        self.loc_adjust = nn.Conv2d(loc_output, loc_output, kernel_size=1)

    def forward(self, z_f, x_f):
        cls_kernel = self.template_cls_conv(z_f)
        loc_kernel = self.template_loc_conv(z_f)

        cls_feature = self.search_cls_conv(x_f)
        loc_feature = self.search_loc_conv(x_f)

        cls = xcorr_fast(cls_feature, cls_kernel)
        loc = self.loc_adjust(xcorr_fast(loc_feature, loc_kernel))
        return cls, loc


class DepthwiseXCorr(nn.Module):
    def __init__(self, in_channels, hidden, out_channels, kernel_size=3):
        super(DepthwiseXCorr, self).__init__()
        self.conv_kernel = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.conv_search = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, out_channels, kernel_size=1)
        )

    def forward(self, kernel, search):
        kernel = self.conv_kernel(kernel)
        search = self.conv_search(search)
        feature = xcorr_depthwise(kernel, search)
        out = self.head(feature)
        return out


class DepthwiseBAN(BAN):
    def __init__(self, in_channels=256, out_channels=256, cls_out_channels=2, weighted=False):
        super(DepthwiseBAN, self).__init__()
        self.cls = DepthwiseXCorr(in_channels, out_channels, cls_out_channels)
        self.loc = DepthwiseXCorr(in_channels, out_channels, 4)

    def forward(self, z_f, x_f):
        cls = self.cls(z_f, x_f)
        loc = self.loc(z_f, x_f)
        return cls, loc


# 多尺度，网络比较重
class MultiBAN(BAN):
    def __init__(self, in_channels, cls_out_channels, weighted=False):
        super(MultiBAN, self).__init__()
        self.weighted = weighted
        for i in range(len(in_channels)):
            # self.add_module('box' + str(i + 2), DepthwiseBAN(in_channels[i], in_channels[i], cls_out_channels))
            self.add_module('box' + str(i + 2), NonLocalBAN(in_channels[i]))
        if self.weighted:
            self.cls_weight = nn.Parameter(torch.ones(len(in_channels)))
            self.loc_weight = nn.Parameter(torch.ones(len(in_channels)))

        self.loc_scale = nn.Parameter(torch.ones(len(in_channels)))

    def forward(self, z_fs, x_fs):
        cls = []
        loc = []

        for idx, (z_f, x_f) in enumerate(zip(z_fs, x_fs), start=2):
            box = getattr(self, 'box' + str(idx))
            c, l = box(z_f, x_f)
            cls.append(c)
            # loc.append(torch.exp(l * self.loc_scale[idx - 2]))
            loc.append(l * self.loc_scale[idx - 2])

        if self.weighted:
            cls_weight = F.softmax(self.cls_weight, 0)
            loc_weight = F.softmax(self.loc_weight, 0)

        def avg(lst):
            return sum(lst) / len(lst)

        def weighted_avg(lst, weight):
            s = 0

            for i in range(len(weight)):
                s += lst[i] * weight[i]
            return s

        if self.weighted:
            return weighted_avg(cls, cls_weight), weighted_avg(loc, loc_weight)
        else:
            return avg(cls), avg(loc)


class NonLocalBAN(nn.Module):
    def __init__(self, in_channels=256):
        super(NonLocalBAN, self).__init__()
        # channel reduce
        self.fi_cls = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.fi_reg = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

        # fdta
        self.task_aware_cls = FDTA(in_channels, size=31)
        self.task_aware_reg = FDTA(in_channels, size=31)

        # predict head
        self.head = PWHead(in_channels)

        # initialization
        for modules in [self.fi_cls, self.fi_reg]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

    def forward(self, zfc, xfc, zfr, xfr):
        # pixel-wise corr
        features_cls = xcorr_pixelwise(zfc, xfc)
        features_reg = xcorr_pixelwise(zfr, xfr)

        # channel reduce
        features_cls = self.fi_cls(features_cls)
        features_reg = self.fi_reg(features_reg)

        # fdta
        features_cls = self.task_aware_cls(features_cls)
        features_reg = self.task_aware_reg(features_reg)

        # predict head
        cls, reg = self.head(features_cls, features_reg)
        return cls, reg
