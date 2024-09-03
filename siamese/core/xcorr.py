# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def xcorr_slow(x, kernel):
    """for loop to calculate cross correlation, slow version
    """
    batch = x.size()[0]
    out = []
    for i in range(batch):
        px = x[i]
        pk = kernel[i]
        px = px.view(1, -1, px.size()[1], px.size()[2])
        pk = pk.view(1, -1, pk.size()[1], pk.size()[2])
        po = F.conv2d(px, pk)
        out.append(po)
    out = torch.cat(out, 0)
    return out


def xcorr_fast(x, kernel):
    """group conv2d to calculate cross correlation, fast version
    """
    batch = kernel.size()[0]
    pk = kernel.view(-1, x.size()[1], kernel.size()[2], kernel.size()[3])
    px = x.view(1, -1, x.size()[2], x.size()[3])
    po = F.conv2d(px, pk, groups=batch)
    po = po.view(batch, -1, po.size()[2], po.size()[3])
    return po


def xcorr_depthwise(kernel, x):
    """depthwise cross correlation
    """
    batch = kernel.size(0)
    channel = kernel.size(1)
    x = x.view(1, batch*channel, x.size(2), x.size(3))
    kernel = kernel.view(batch*channel, 1, kernel.size(2), kernel.size(3))
    out = F.conv2d(x, kernel, groups=batch*channel)
    out = out.view(batch, channel, out.size(2), out.size(3))
    return out


def xcorr_pixelwise(fm, fq):
    """
    fm: z (kernel)
    fq: x
    """
    B, C, h, w = fm.shape
    B, C, H, W = fq.shape

    fm0 = fm.clone()
    fq0 = fq.clone()

    fm = fm.contiguous().view(B, C, h * w)  # B, C, hw
    fm = fm.permute(0, 2, 1)  # B, hw, C
    fq = fq.contiguous().view(B, C, H * W)  # B, C, HW

    similar = torch.matmul(fm, fq) / math.sqrt(C)  # B, hw, HW
    similar = torch.softmax(similar, dim=1)  # B, hw, HW

    fm1 = fm0.view(B, C, h * w)  # B, C, hw
    mem_info = torch.matmul(fm1, similar)  # (B, C, hw) x (B, hw, HW) = (B, C, HW)
    mem_info = mem_info.view(B, C, H, W)

    y = torch.cat([mem_info, fq0], dim=1)
    return y
