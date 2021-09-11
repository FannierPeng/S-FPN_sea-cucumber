# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import os
#改root path

ROOT_PATH = os.path.abspath('..')
# pretrain weights path
TEST_SAVE_PATH = ROOT_PATH + r'/tools/test_result'
# INFERENCE_IMAGE_PATH = os.path.abspath('../..')+ r'/DataSet/sea_cucumber/test/JPEGImages'
INFERENCE_IMAGE_PATH = ROOT_PATH +r'tools/inference_result/videos'
INFERENCE_SAVE_PATH = ROOT_PATH + r'tools/inference_result'
#重点改
NET_NAME = 'resnet_v1_50'
# NET_NAME = 'mobilenet_224'
# NET_NAME = 'inception_v4'
VERSION = 'v39_sea cucumber'
CLASS_NUM = 1
BASE_ANCHOR_SIZE_LIST = [15, 25, 40, 60, 80]#我们的
# BASE_ANCHOR_SIZE_LIST = [32, 64, 128, 256, 512]#voc的
# BASE_ANCHOR_SIZE_LIST = [10, 20, 35, 50, 70]
LEVEL = ['P2', 'P3', 'P4', 'P5', "P6"]
STRIDE = [4, 8, 16, 32, 64]
# ANCHOR_SCALES = [2 ** - 3,2 ** - 2,2 ** - 1]
ANCHOR_SCALES = [2**-3,2**-2,4,8,12]#ours最佳
# ANCHOR_SCALES = [1.0]
# ANCHOR_SCALES = [2**-3,2**-2,3.47,7.24, 12.58]#kmeans[]
# ANCHOR_SCALES = [1.81,3.76,5.66,7.99, 11.17]#kmeans[]
# ANCHOR_SCALES=[2 ** - 2,2 ** - 1,1]
# ANCHOR_RATIOS = [1, 0.5, 2, 1.5, 1 / 1.5, 3.5, 1/3.5, 2.8, 1 / 2.8]
ANCHOR_RATIOS = [1., 0.5, 2., 1. / 1.5, 1.5, 1. / 3., 3.]#我们的

# ANCHOR_RATIOS = [0.788, 2.12, 5.29, 1.47, 2.98]
# ANCHOR_RATIOS = [1.65,3.2 ,5.29,0.64,2.27,1.14]
# ANCHOR_RATIOS = [0.57, 1.01, 1.47, 1.92, 2.4, 2.98, 3.99]

SCALE_FACTORS = [10., 10., 5., 5.]
OUTPUT_STRIDE = 16
# OUTPUT_STRIDE = 8
SHORT_SIDE_LEN = 600

DATASET_NAME = 'sea cucumber'

BATCH_SIZE = 1
WEIGHT_DECAY = {'resnet_v1_50': 0.0001, 'resnet_v1_101': 0.0001,'pvanet': 0.0001, 'vgg_16':0.0005, 'inception_resnet':0.00004, 'inception_v4':0.00004, 'mobilenet_224': 0.00004}
EPSILON = 1e-5
MOMENTUM = 0.9
MAX_ITERATION = 40000
GPU_GROUP='0,1'
LR = 10**-4
#LR=0.00002
#LR = 0.001

# rpn
SHARE_HEAD = True
RPN_NMS_IOU_THRESHOLD = 0.5
MAX_PROPOSAL_NUM = 2000
RPN_IOU_POSITIVE_THRESHOLD = 0.5
RPN_IOU_NEGATIVE_THRESHOLD = 0.2
RPN_MINIBATCH_SIZE = 512
# RPN_MINIBATCH_SIZE = 256
RPN_POSITIVE_RATE = 0.5
IS_FILTER_OUTSIDE_BOXES = True
RPN_TOP_K_NMS = 12000

# fast rcnn
ROI_SIZE = 14
ROI_POOL_KERNEL_SIZE = 2
USE_DROPOUT = False
KEEP_PROB = 0.5
FAST_RCNN_NMS_IOU_THRESHOLD = 0.2
FAST_RCNN_NMS_MAX_BOXES_PER_CLASS = 100
FINAL_SCORE_THRESHOLD = 0.5
# FINAL_SCORE_THRESHOLD = 0.75
FAST_RCNN_IOU_POSITIVE_THRESHOLD = 0.45
FAST_RCNN_MINIBATCH_SIZE = 256
FAST_RCNN_POSITIVE_RATE = 0.5

