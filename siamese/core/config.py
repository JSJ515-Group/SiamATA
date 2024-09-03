# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from yacs.config import CfgNode as CN

__C = CN()

cfg = __C

__C.META_ARC = "resnet50_all"

__C.CUDA = True

# ------------------------------------------------------------------------ #
# Training options
# ------------------------------------------------------------------------ #
__C.TRAIN = CN()

# Anchor Target
__C.TRAIN.POINT = 'normal'

__C.TRAIN.REG_MAX = 32
__C.TRAIN.REG_TOPK = 4
__C.TRAIN.IOU = True
__C.TRAIN.IOUGUIDED = True

__C.TRAIN.EXEMPLAR_SIZE = 127

__C.TRAIN.SEARCH_SIZE = 255

__C.TRAIN.BASE_SIZE = 8

__C.TRAIN.OUTPUT_SIZE = 25

__C.TRAIN.RESUME = ''

__C.TRAIN.PRETRAINED = ''

__C.TRAIN.LOG_DIR = './logs'

__C.TRAIN.SNAPSHOT_DIR = './snapshot'

__C.TRAIN.EPOCH = 20

__C.TRAIN.START_EPOCH = 0

__C.TRAIN.BATCH_SIZE = 32

__C.TRAIN.NUM_WORKERS = 14

__C.TRAIN.MOMENTUM = 0.9

__C.TRAIN.WEIGHT_DECAY = 0.0001

__C.TRAIN.CLS_WEIGHT = 1.0

__C.TRAIN.LOC_WEIGHT = 2.0

__C.TRAIN.DFL_WEIGHT = 1.0

__C.TRAIN.PRINT_FREQ = 20

__C.TRAIN.LOG_GRADS = False

__C.TRAIN.GRAD_CLIP = 10.0

__C.TRAIN.BASE_LR = 0.005

__C.TRAIN.LR = CN()

__C.TRAIN.LR.TYPE = 'log'

__C.TRAIN.LR.KWARGS = CN(new_allowed=True)

__C.TRAIN.LR_WARMUP = CN()

__C.TRAIN.LR_WARMUP.WARMUP = True

__C.TRAIN.LR_WARMUP.TYPE = 'step'

__C.TRAIN.LR_WARMUP.EPOCH = 5

__C.TRAIN.LR_WARMUP.KWARGS = CN(new_allowed=True)

__C.TRAIN.NUM_CLASSES = 2

__C.TRAIN.NUM_CONVS = 4

__C.TRAIN.PRIOR_PROB = 0.01

__C.TRAIN.LOSS_ALPHA = 0.25

__C.TRAIN.LOSS_GAMMA = 2.0

# ------------------------------------------------------------------------ #
# Dataset options
# ------------------------------------------------------------------------ #
__C.DATASET = CN(new_allowed=True)

# Augmentation
# for template
__C.DATASET.TEMPLATE = CN()

# Random shift see [SiamPRN++](https://arxiv.org/pdf/1812.11703)
# for detail discussion
__C.DATASET.TEMPLATE.SHIFT = 4

__C.DATASET.TEMPLATE.SCALE = 0.05

__C.DATASET.TEMPLATE.BLUR = 0.0

__C.DATASET.TEMPLATE.FLIP = 0.0

__C.DATASET.TEMPLATE.COLOR = 1.0

__C.DATASET.SEARCH = CN()

__C.DATASET.SEARCH.SHIFT = 64

__C.DATASET.SEARCH.SCALE = 0.18

__C.DATASET.SEARCH.BLUR = 0.0

__C.DATASET.SEARCH.FLIP = 0.0

__C.DATASET.SEARCH.COLOR = 1.0

# Sample Negative pair see [DaSiamRPN](https://arxiv.org/pdf/1808.06048)
# for detail discussion
__C.DATASET.NEG = 0.0

# improve tracking performance for otb100
__C.DATASET.GRAY = 0.0

__C.DATASET.NAMES = ('VID', 'YOUTUBEBB', 'DET', 'COCO', 'GOT10K', 'LASOT')

__C.DATASET.VID = CN()
__C.DATASET.VID.ROOT = '/home/yxy/xyz/training_dataset/vid/crop511'
__C.DATASET.VID.ANNO = '/home/yxy/xyz/training_dataset/vid/train.json'
__C.DATASET.VID.FRAME_RANGE = 100
__C.DATASET.VID.NUM_USE = 100000

__C.DATASET.YOUTUBEBB = CN()
__C.DATASET.YOUTUBEBB.ROOT = '/home/yxy/xyz/training_dataset/yt_bb/crop511'
__C.DATASET.YOUTUBEBB.ANNO = '/home/yxy/xyz/training_dataset/yt_bb/train.json'
__C.DATASET.YOUTUBEBB.FRAME_RANGE = 3
__C.DATASET.YOUTUBEBB.NUM_USE = -1

__C.DATASET.COCO = CN()
__C.DATASET.COCO.ROOT = '/home/yxy/xyz/training_dataset/coco/crop511'
__C.DATASET.COCO.ANNO = '/home/yxy/xyz/training_dataset/coco/train2017.json'
__C.DATASET.COCO.FRAME_RANGE = 1
__C.DATASET.COCO.NUM_USE = -1

__C.DATASET.DET = CN()
__C.DATASET.DET.ROOT = '/home/yxy/xyz/training_dataset/det/crop511'
__C.DATASET.DET.ANNO = '/home/yxy/xyz/training_dataset/det/train.json'
__C.DATASET.DET.FRAME_RANGE = 1
__C.DATASET.DET.NUM_USE = -1

__C.DATASET.GOT10K = CN()
__C.DATASET.GOT10K.ROOT = '/home/yxy/xyz/training_dataset/got10k/crop511'
__C.DATASET.GOT10K.ANNO = '/home/yxy/xyz/training_dataset/got10k/train.json'
__C.DATASET.GOT10K.FRAME_RANGE = 50
__C.DATASET.GOT10K.NUM_USE = 100000

__C.DATASET.LASOT = CN()
__C.DATASET.LASOT.ROOT = '/home/yxy/xyz/training_dataset/lasot/crop511'
__C.DATASET.LASOT.ANNO = '/home/yxy/xyz/training_dataset/lasot/train.json'
__C.DATASET.LASOT.FRAME_RANGE = 100
__C.DATASET.LASOT.NUM_USE = 100000

__C.DATASET.VIDEOS_PER_EPOCH = 600000
# ------------------------------------------------------------------------ #
# Backbone options
# ------------------------------------------------------------------------ #
__C.BACKBONE = CN()

# Backbone type, current only support resnet18,34,50;alexnet;mobilenet
__C.BACKBONE.TYPE = 'res50'

__C.BACKBONE.KWARGS = CN(new_allowed=True)

# Pretrained backbone weights
__C.BACKBONE.PRETRAINED = ''

# Train layers
__C.BACKBONE.TRAIN_LAYERS = ['layer1', 'layer2', 'layer3']

# Layer LR
__C.BACKBONE.LAYERS_LR = 0.1

# Switch to train layer
__C.BACKBONE.TRAIN_EPOCH = 10

# ------------------------------------------------------------------------ #
# Adjust layer options
# ------------------------------------------------------------------------ #
__C.ADJUST = CN()

# Adjust layer
__C.ADJUST.ADJUST = True

__C.ADJUST.KWARGS = CN(new_allowed=True)

# Adjust layer type
__C.ADJUST.TYPE = "AdjustAllLayer"

# ------------------------------------------------------------------------ #
# BAN options
# ------------------------------------------------------------------------ #
__C.BAN = CN()

# Whether to use ban head
__C.BAN.BAN = False

# BAN type
__C.BAN.TYPE = 'MultiBAN'

__C.BAN.KWARGS = CN(new_allowed=True)


# ------------------------------------------------------------------------ #
# Tracker options
# ------------------------------------------------------------------------ #
__C.TRACK = CN()

__C.TRACK.TYPE = 'SiamBANTracker'

# Scale penalty
__C.TRACK.PENALTY_K = 0.04

__C.TRACK.INTERPOLATION = False

# Window influence
__C.TRACK.WINDOW_INFLUENCE = 0.44

# Interpolation learning rate
__C.TRACK.LR = 0.4

# Exemplar size
__C.TRACK.EXEMPLAR_SIZE = 127

# Instance size
__C.TRACK.INSTANCE_SIZE = 255

# Base size
__C.TRACK.BASE_SIZE = 8

# Context amount
__C.TRACK.CONTEXT_AMOUNT = 0.5

__C.TRACK.STRIDE = 8.0

__C.TRACK.SCORE_SIZE = 25

__C.TRACK.hanming = True

__C.TRACK.NUM_K = 2

__C.TRACK.NUM_N = 1

__C.TRACK.REGION_S = 0.1

__C.TRACK.REGION_L = 0.44


# ------------------------------------------------------------------------ #
# HP_SEARCH parameters
# ------------------------------------------------------------------------ #
__C.HP_SEARCH = CN()

__C.HP_SEARCH.UAV123 = [0.2390650707807188, 0.036010901145472576, 0.19817652920292675]

__C.HP_SEARCH.OTB100 = [0.6037291666153346, 0.20484205699557875, 0.5824052956484521]

__C.HP_SEARCH.VOT2018 = [0.5003972357626941, 0.15761085296157645, 0.34783423829042132]

__C.HP_SEARCH.GOT10K = [0.9664466462859548, 0.176729998657605, 0.2527547253394025]

__C.HP_SEARCH.LASOT = [0.87748401336284709, 0.14794071292069522, 0.17530970310568331]

__C.HP_SEARCH.VOT2016 = [0.5003972357626941, 0.15761085296157645, 0.34783423829042132]

__C.HP_SEARCH.VOT2019 = [0.5003972357626941, 0.15761085296157645, 0.34783423829042132]
