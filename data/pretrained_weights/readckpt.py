#!/user/bin/env python
# _*_ coding:utf-8 _*_
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

reader = tf.train.NewCheckpointReader("E:\commonly_codes\FPN_Tensorflow-yangxue-sea-cucumber\data\pretrained_weights/pva9.1_pretrained_no_fc6.ckpt")

variables = reader.get_variable_to_shape_map()

for ele in variables:
    print(ele)
