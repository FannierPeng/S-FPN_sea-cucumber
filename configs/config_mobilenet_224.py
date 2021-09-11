#!/user/bin/env python
# _*_ coding:utf-8 _*_
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

tf.app.flags.DEFINE_string(
    'dataset_tfrecord',
    '../data/tfrecords',
    'tfrecord of fruits dataset'
)
tf.app.flags.DEFINE_integer(
    'shortside_size',
    224,
    'the value of new height and new width, new_height = new_width'
)

###########################
#  data batch
##########################
tf.app.flags.DEFINE_integer(
    'num_classes',
    1,
    'num of classes'
)
tf.app.flags.DEFINE_integer(
    'batch_size',
    1, #64
    'num of imgs in a batch'
)

###############################
# optimizer-- MomentumOptimizer
###############################
tf.app.flags.DEFINE_float(
    'momentum',
    0.9,
    'accumulation = momentum * accumulation + gradient'
)

############################
#  train
########################
tf.app.flags.DEFINE_integer(
    'max_steps',
    900000,
    'max iterate steps'
)

# tf.app.flags.DEFINE_string(
#     'pretrained_model_path',
#     '../data/pretrained_weights/pva9.1_preAct_train_iter_1900000.npy',
#     'the path of pretrained weights'
# )
tf.app.flags.DEFINE_string(
    'pretrained_model_path',
    '../data/pretrained_weights/mobilenet_v1_1.0_224/mobilenet_v1_1.0_224.ckpt',
    'the path of pretrained weights'
)
tf.app.flags.DEFINE_float(
    'weight_decay',
    0.00004,
    'weight_decay in regulation'
)
################################
# summary and save_weights_checkpoint
##################################
tf.app.flags.DEFINE_string(
    'summary_path',
    '../output/mobilenet_summary',
    'the path of summary write to '
)
tf.app.flags.DEFINE_string(
    'trained_checkpoint',
    '../output/mobilenet_trained_weights',
    'the path to save trained_weights'
)
FLAGS = tf.app.flags.FLAGS