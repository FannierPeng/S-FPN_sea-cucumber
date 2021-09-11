# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

from libs.configs import cfgs

if cfgs.DATASET_NAME == 'ship':
    NAME_LABEL_MAP = {
        'back_ground': 0,
        "ship": 1
    }
elif cfgs.DATASET_NAME == 'SSDD':
    NAME_LABEL_MAP = {
        'back_ground': 0,
        "ship": 1
    }
elif cfgs.DATASET_NAME == 'airplane':
    NAME_LABEL_MAP = {
        'back_ground': 0,
        "airplane": 1
    }
elif cfgs.DATASET_NAME == 'sea cucumber':
    NAME_LABEL_MAP = {
        'back_ground': 0,
        "sea cucumber": 1
    }
elif cfgs.DATASET_NAME == 'nwpu':
    NAME_LABEL_MAP = {
        'back_ground': 0,
        'airplane': 1,
        'ship': 2,
        'storage tank': 3,
        'baseball diamond': 4,
        'tennis court': 5,
        'basketball court': 6,
        'ground track field': 7,
        'harbor': 8,
        'bridge': 9,
        'vehicle': 10,
    }
elif cfgs.DATASET_NAME == 'pascal':
    NAME_LABEL_MAP = {
        'back_ground': 0,
        'aeroplane': 1,
        'bicycle': 2,
        'bird': 3,
        'boat': 4,
        'bottle': 5,
        'bus': 6,
        'car': 7,
        'cat': 8,
        'chair': 9,
        'cow': 10,
        'diningtable': 11,
        'dog': 12,
        'horse': 13,
        'motorbike': 14,
        'person': 15,
        'pottedplant': 16,
        'sheep': 17,
        'sofa': 18,
        'train': 19,
        'tvmonitor': 20
    }
else:
    assert 'please set label dict!'

def get_label_name_map():
    reverse_dict = {}
    for name, label in NAME_LABEL_MAP.items():
        reverse_dict[label] = name
    return reverse_dict

LABEl_NAME_MAP = get_label_name_map()