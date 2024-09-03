# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch.nn as nn

from siamese.core.config import cfg
from siamese.models.backbone import get_backbone
from siamese.models.head import get_ban_head
from siamese.models.neck import get_neck
from siamese.models.loss.loss_main import make_loss_evaluator, Project
from siamese.utils.location_grid import compute_locations


class ModelBuilder(nn.Module):
    def __init__(self):
        super(ModelBuilder, self).__init__()

        # build backbone
        self.backbone = get_backbone(cfg.BACKBONE.TYPE,
                                     **cfg.BACKBONE.KWARGS)

        # build adjust layer
        # 4 neck for Asymmetric Structure of siamese networks
        if cfg.ADJUST.ADJUST:
            self.neck_ct = get_neck(cfg.ADJUST.TYPE,
                                    **cfg.ADJUST.KWARGS)
            self.neck_cs = get_neck(cfg.ADJUST.TYPE,
                                    **cfg.ADJUST.KWARGS)
            self.neck_rt = get_neck(cfg.ADJUST.TYPE,
                                    **cfg.ADJUST.KWARGS)
            self.neck_rs = get_neck(cfg.ADJUST.TYPE,
                                    **cfg.ADJUST.KWARGS)

        # build ban head
        if cfg.BAN.BAN:
            self.head = get_ban_head(cfg.BAN.TYPE,
                                     **cfg.BAN.KWARGS)

        # build loss
        self.loss_evaluator = make_loss_evaluator(cfg)

        # just for DFL
        self.reg_max = cfg.TRAIN.REG_MAX
        if self.reg_max > 0:
            self.distribution_project = Project(self.reg_max)

    def template(self, z):
        zf = self.backbone(z)
        if cfg.ADJUST.ADJUST:
            zfc = self.neck_ct(zf)
            zfr = self.neck_rt(zf)
        self.zfc = zfc
        self.zfr = zfr

    def track(self, x):
        xf = self.backbone(x)
        if cfg.ADJUST.ADJUST:
            xfc = self.neck_cs(xf)
            xfr = self.neck_rs(xf)
        cls, loc = self.head(self.zfc, xfc, self.zfr, xfr)
        if self.reg_max > 0:
            b, c, h, w = cls.shape
            loc = loc.permute(0, 2, 3, 1).contiguous().view(-1, self.reg_max + 1)
            loc = self.distribution_project(loc)
            loc = loc.view(b, w, h, 4).permute(0, 3, 1, 2).contiguous() * cfg.TRACK.STRIDE
        return {
            'cls': cls,
            'loc': loc
        }

    def cls_aixs(self, cls):
        cls = cls.permute(0, 2, 3, 1).contiguous()
        return cls

    def forward(self, data):
        """ only used in training
        """
        template = data['template'].cuda()
        search = data['search'].cuda()
        label_cls = data['label_cls'].cuda()
        label_loc = data['bbox'].cuda()

        # get feature
        zf = self.backbone(template)
        xf = self.backbone(search)
        if cfg.ADJUST.ADJUST:
            zfc = self.neck_ct(zf)
            zfr = self.neck_rt(zf)
            xfc = self.neck_cs(xf)
            xfr = self.neck_rs(xf)
        cls, loc = self.head(zfc, xfc, zfr, xfr)

        # get loss
        locations = compute_locations(cls, cfg.TRACK.STRIDE)
        cls = self.cls_aixs(cls)

        # use dfl
        if self.reg_max > 0:
            cls_loss, loc_loss, dfl_loss = self.loss_evaluator(locations, cls, loc, label_cls, label_loc)
        # no dfl
        else:
            cls_loss, loc_loss = self.loss_evaluator(locations, cls, loc, label_cls, label_loc)

        outputs = {}
        outputs['total_loss'] = cfg.TRAIN.CLS_WEIGHT * cls_loss + cfg.TRAIN.LOC_WEIGHT * loc_loss
        outputs['cls_loss'] = cls_loss
        outputs['loc_loss'] = loc_loss
        if self.reg_max > 0:
            outputs['total_loss'] += cfg.TRAIN.DFL_WEIGHT * dfl_loss
            outputs['dfl_loss'] = dfl_loss

        return outputs
