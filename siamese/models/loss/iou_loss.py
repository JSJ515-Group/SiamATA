import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from siamese.core.config import cfg

"""
yolo中iou和本文iou对应
yolo采用的是(x, y, w, h)，表示边界框，x,y表示边界框中心点坐标，w,h分别为边界框的宽高
    w1 = pred_left + pred_right
    h1 = pred_top + pred_bottom
    w2 = target_left + target_right
    h2 = target_top + target_bottom
    
    pred_left = b1_x1
    pred_right = b1_x2
    pred_bottom = b1_y1
    pred_top = b1_y2
    
    target_left = b2_x1
    target_right = b2_x2
    target_bottom = b2_y1
    target_top = b2_y2
"""


# get from IOULoss
def cal_iou(target_left, target_top, target_right, target_bottom,
            pred_left, pred_top, pred_right, pred_bottom):
    # GT边界框面积
    target_aera = (target_left + target_right) * \
                  (target_top + target_bottom)
    # 预测边界框面积
    pred_aera = (pred_left + pred_right) * \
                (pred_top + pred_bottom)

    # w_intersect和h_intersect: 这两个变量分别计算预测边界框和目标边界框在宽度和高度方向上的交集。
    w_intersect = torch.min(pred_left, target_left) + torch.min(pred_right, target_right)
    h_intersect = torch.min(pred_bottom, target_bottom) + torch.min(pred_top, target_top)

    # 计算预测边界框和目标边界框的交集面积
    area_intersect = w_intersect * h_intersect
    # 计算预测边界框和目标边界框的并集面积
    area_union = target_aera + pred_aera - area_intersect
    eps = area_union.new_tensor([1e-5])
    area_union = torch.max(area_union, eps)
    # 计算交并比
    iou = area_intersect / area_union
    return iou


# get from yolo
def cal_iou_new(target_left, target_top, target_right, target_bottom,
                pred_left, pred_top, pred_right, pred_bottom, eps):
    w1 = pred_left + pred_right
    h1 = pred_top + pred_bottom
    w2 = target_left + target_right
    h2 = target_top + target_bottom

    inter = ((torch.min(pred_right, target_right) - torch.max(pred_left, target_left)).clamp(0) *
             (torch.min(pred_top, target_top) - torch.max(pred_bottom, target_bottom)).clamp(0))
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union
    return iou


# common iou
class IOULoss(nn.Module):
    def __init__(self, name="iou", gama=0.5):
        super(IOULoss, self).__init__()
        self.name = name
        self.gama = gama

    def forward(self, pred, target, weight=None):
        if cfg.TRAIN.REG_MAX > 0:
            pred = pred * cfg.TRACK.STRIDE
            target = target * cfg.TRACK.STRIDE
        # 预测边界框四个边界坐标，坐标点而不是偏移量，可以从下面中心点坐标看出来
        pred_left = pred[:, 0]
        pred_top = pred[:, 1]
        pred_right = pred[:, 2]
        pred_bottom = pred[:, 3]

        # GT边界框四个边界坐标
        target_left = target[:, 0]
        target_top = target[:, 1]
        target_right = target[:, 2]
        target_bottom = target[:, 3]

        # GT边界框面积
        target_aera = (target_left + target_right) * \
                      (target_top + target_bottom)
        # 预测边界框面积
        pred_aera = (pred_left + pred_right) * \
                    (pred_top + pred_bottom)

        # w_intersect和h_intersect: 这两个变量分别计算预测边界框和目标边界框在宽度和高度方向上的交集。
        w_intersect = torch.min(pred_left, target_left) + torch.min(pred_right, target_right)
        h_intersect = torch.min(pred_bottom, target_bottom) + torch.min(pred_top, target_top)

        # 计算预测边界框和目标边界框的交集面积
        area_intersect = w_intersect * h_intersect
        # 计算预测边界框和目标边界框的并集面积
        area_union = target_aera + pred_aera - area_intersect
        eps = area_union.new_tensor([1e-5])
        area_union = torch.max(area_union, eps)
        # 计算交并比
        iou = area_intersect / area_union

        # 　计算包围所有边界框的最小矩形的坐标、宽高和面积
        enclose_x1y1 = torch.max(pred[:, :2], target[:, :2])
        enclose_x2y2 = torch.max(pred[:, 2:], target[:, 2:])
        enclose_wh = (enclose_x2y2 + enclose_x1y1).clamp(min=0)
        enclose_area = enclose_wh[:, 0] * enclose_wh[:, 1]

        # 计算预测边界框和目标边界框的中心点坐标
        predcenter_x = (pred_right - pred_left) / 2.0
        predcenter_y = (pred_top - pred_bottom) / 2.0
        targetcenter_x = (target_right - target_left) / 2.0
        targetcenter_y = (target_top - target_bottom) / 2.0

        # 计算预测边界框和目标边界框中心点之间的距离的平方
        inter_diag = (predcenter_x - targetcenter_x) ** 2 + (predcenter_y - targetcenter_y) ** 2

        # 计算包围所有边界框的最小矩形的对角线长度的平方
        outer_diag = enclose_wh[:, 0] ** 2 + enclose_wh[:, 1] ** 2

        if self.name == "logiou":
            losses = - torch.log(iou)
        elif self.name == "giou":
            gious = iou - (enclose_area - area_union) / enclose_area
            losses = 1 - gious
        elif "diou" in self.name:
            losses = 1 - iou + inter_diag / outer_diag
            if "log" in self.name:
                losses = - torch.log(1 - losses / 2)
        elif "ciou" in self.name:
            w2 = target_left + target_right
            h2 = target_top + target_bottom
            w1 = pred_left + pred_right
            h1 = pred_top + pred_bottom
            v = (4 / (math.pi ** 2)) * torch.pow((torch.atan(w2 / h2) - torch.atan(w1 / h1)), 2)
            S = 1 - iou
            k = torch.max(S + v, eps)
            alpha = v / k
            losses = 1 - iou + inter_diag / outer_diag + alpha * v
            if "log" in self.name:
                losses = - torch.log(1 - losses / 3)
        elif "eiou" in self.name:
            lw = (target_left + target_right - (pred_left + pred_right)) ** 2.0
            lh = (target_top + target_bottom - (pred_top + pred_bottom)) ** 2.0
            outer_diag = torch.max(outer_diag, eps)
            enclose_w = torch.max(enclose_wh[:, 0], eps)
            enclose_h = torch.max(enclose_wh[:, 1], eps)
            losses = 1 - iou + inter_diag / outer_diag + lw / enclose_w ** 2 + lh / enclose_h ** 2
            if "f" in self.name or "F" in self.name:
                losses = losses * (iou ** self.gama)
            if "log" in self.name:
                losses = - torch.log(1 - losses / 4)
        else:
            losses = 1 - iou

        if weight is not None and weight.sum() > 0:
            return (losses * weight).sum() / weight.sum()
        else:
            assert losses.numel() != 0
            return losses.mean()


class IoULossCollection(nn.Module):
    def __init__(self, name="shapeiou", use_inner=False, use_log=False):
        super(IoULossCollection, self).__init__()
        self.name = name
        self.use_inner = use_inner
        self.use_log = use_log

    def forward(self, pred, target, ratio=0.7, scale=0.0, d=0.00, u=0.95, eps=1e-7, weight=None):
        if cfg.TRAIN.REG_MAX > 0:
            pred = pred * cfg.TRACK.STRIDE
            target = target * cfg.TRACK.STRIDE

        # iou
        pred_left = pred[:, 0]
        pred_top = pred[:, 1]
        pred_right = pred[:, 2]
        pred_bottom = pred[:, 3]

        target_left = target[:, 0]
        target_top = target[:, 1]
        target_right = target[:, 2]
        target_bottom = target[:, 3]

        # use inner (ider from inner-iou)
        if self.use_inner:
            inner_pred_left = pred_left * ratio
            inner_pred_top = pred_top * ratio
            inner_pred_right = pred_right * ratio
            inner_pred_bottom = pred_bottom * ratio

            inner_target_left = target_left * ratio
            inner_target_top = target_top * ratio
            inner_target_right = target_right * ratio
            inner_target_bottom = target_bottom * ratio

            inner_iou = cal_iou_new(inner_target_left, inner_target_top, inner_target_right, inner_target_bottom,
                                    inner_pred_left, inner_pred_top, inner_pred_right, inner_pred_bottom, eps)

            iou = inner_iou
        # normal iou
        else:
            iou = cal_iou_new(target_left, target_top, target_right, target_bottom,
                              pred_left, pred_top, pred_right, pred_bottom, eps)

        # 计算包围所有边界框的最小矩形的坐标、宽高和面积
        enclose_x1y1 = torch.max(pred[:, :2], target[:, :2])
        enclose_x2y2 = torch.max(pred[:, 2:], target[:, 2:])
        enclose_wh = (enclose_x2y2 + enclose_x1y1).clamp(min=0)

        # 计算预测边界框和目标边界框的中心点坐标
        predcenter_x = (pred_right - pred_left) / 2.0
        predcenter_y = (pred_top - pred_bottom) / 2.0
        targetcenter_x = (target_right - target_left) / 2.0
        targetcenter_y = (target_top - target_bottom) / 2.0

        # 计算预测边界框和目标边界框中心点之间的距离的平方
        inter_diag = (predcenter_x - targetcenter_x) ** 2 + (predcenter_y - targetcenter_y) ** 2
        # 计算包围所有边界框的最小矩形的对角线长度的平方
        outer_diag = enclose_wh[:, 0] ** 2 + enclose_wh[:, 1] ** 2

        # 所有边界框的宽、高
        w1 = pred_left + pred_right
        h1 = pred_top + pred_bottom
        w2 = target_left + target_right
        h2 = target_top + target_bottom

        if "focaleriou" in self.name:
            focaler_iou = ((iou - d) / (u - d)).clamp(0, 1)  # default d=0.00,u=0.95
            losses = 1 - focaler_iou
        elif "shapeiou" in self.name:
            # shape iou
            ww = 2 * torch.pow(w2, scale) / (torch.pow(w2, scale) + torch.pow(h2, scale))
            hh = 2 * torch.pow(h2, scale) / (torch.pow(w2, scale) + torch.pow(h2, scale))

            cw = torch.max(pred_right, target_right) - torch.min(pred_left, target_left)  # convex width
            ch = torch.max(pred_top, target_top) - torch.min(pred_bottom, target_bottom)  # convex height
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            center_distance_x = ((target_left + target_right - pred_left - pred_right) ** 2) / 4
            center_distance_y = ((target_bottom + target_top - pred_bottom - pred_top) ** 2) / 4
            center_distance = hh * center_distance_x + ww * center_distance_y
            distance = center_distance / c2

            # Shape-Shape
            omiga_w = hh * torch.abs(w1 - w2) / torch.max(w1, w2)
            omiga_h = ww * torch.abs(h1 - h2) / torch.max(h1, h2)
            shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + torch.pow(1 - torch.exp(-1 * omiga_h), 4)

            # Shape-IoU
            shape_iou = iou - distance - 0.5 * shape_cost
            losses = 1 - shape_iou
        elif "innerciou" in self.name:
            # inner ciou
            v = (4 / (math.pi ** 2)) * torch.pow((torch.atan(w2 / h2) - torch.atan(w1 / h1)), 2)
            with torch.no_grad():
                alpha = v / (v - iou + (1 + eps))
            inner_ciou = inner_iou - (inter_diag / outer_diag + v * alpha)
            losses = 1 - inner_ciou
            if self.use_log:
                losses = - torch.log(1 - losses / 3)
        elif self.name == "logiou":
            losses = - torch.log(iou)

        # 采用分类分支特征图进行加权
        if weight is not None and weight.sum() > 0:
            return (losses * weight).sum() / weight.sum()
        else:
            assert losses.numel() != 0
            return losses.mean()
