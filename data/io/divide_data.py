# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import sys

sys.path.append('../../')
import shutil
import os
import random
import math
import sys
import time
from libs.configs import cfgs


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

divide_rate = 0.5

#重点改
#数据的根路径
root_path = cfgs.ROOT_PATH +'/sea cucumber/'

image_path = root_path + 'val/test/JPEGImages'
xml_path = root_path + 'val/test/Annotations'

image_list = os.listdir(image_path)

image_name = [n.split('.')[0] for n in image_list]

random.shuffle(image_name)
random.shuffle(image_name)
random.shuffle(image_name)

train_image = image_name[:int(math.ceil(len(image_name)) * divide_rate)]
test_image = image_name[int(math.ceil(len(image_name)) * divide_rate):]

image_output_train = os.path.join(root_path, 'val/test/val/JPEGImages')
mkdir(image_output_train)
image_output_test = os.path.join(root_path, 'val/test/test/JPEGImages')
mkdir(image_output_test)

xml_train = os.path.join(root_path, 'val/test/val/Annotations')
mkdir(xml_train)
xml_test = os.path.join(root_path, 'val/test/test/Annotations')
mkdir(xml_test)

index = 0
for i in train_image:
    shutil.copy(os.path.join(image_path, i + '.jpg'), image_output_train)
    shutil.copy(os.path.join(xml_path, i + '.xml'), xml_train)
    sys.stdout.write('\r>> Converting image %d/%d' % (index + 1, len(train_image)))
    sys.stdout.flush()
    time.sleep(0.2)
    index += 1

index = 0
for i in test_image:
    shutil.copy(os.path.join(image_path, i + '.jpg'), image_output_test)
    shutil.copy(os.path.join(xml_path, i + '.xml'), xml_test)

    sys.stdout.write('\r>> Converting image %d/%d' % (index + 1, len(test_image)))
    sys.stdout.flush()
    time.sleep(0.2)
    index += 1








