from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
import tensorflow.contrib.slim as slim

import os,sys
sys.path.insert(0, '../../')

from libs.networks.slim_nets import resnet_v1
from libs.networks.slim_nets import mobilenet_v1
from libs.networks.slim_nets import inception_resnet_v2, inception_v4
from libs.networks.slim_nets import vgg
from libs.networks.slim_nets import pvanet
# FLAGS = get_flags_byname()


def get_flags_byname(net_name):
    if net_name not in ['resnet_v1_50', 'mobilenet_224', 'inception_resnet', 'vgg_16', 'resnet_v1_101','pvanet','inception_v4']:
        raise ValueError("not include network: {}, we allow resnet_v1_50, mobilenet_224, inception_resnet, "
                         "vgg16, resnet_v1_101"
                         "")

    if net_name == 'resnet_v1_50':
        from configs import config_resnet_50
        return config_resnet_50.FLAGS
    if net_name == 'mobilenet_224':
        from configs import config_mobilenet_224
        return config_mobilenet_224.FLAGS
    if net_name == 'inception_resnet':
        from configs import config_inception_resnet
        return config_inception_resnet.FLAGS
    if net_name == 'vgg_16':
        from configs import config_vgg16
        return config_vgg16.FLAGS
    if net_name == 'resnet_v1_101':
        from configs import config_res101
        return config_res101.FLAGS
    if net_name == 'pvanet':
        from configs import config_pvanet
        return config_pvanet.FLAGS
    if net_name == 'inception_v4':
        from configs import config_inception
        return config_inception.FLAGS

def get_network_byname(net_name,
                       inputs,
                       num_classes=None,
                       is_training=True,
                       global_pool=True,
                       output_stride=None,
                       spatial_squeeze=True):
    if net_name == 'resnet_v1_50':
        FLAGS = get_flags_byname(net_name)
        with slim.arg_scope(resnet_v1.resnet_arg_scope(weight_decay=FLAGS.weight_decay)):
            logits, end_points = resnet_v1.resnet_v1_50(inputs=inputs,
                                                        num_classes=num_classes,
                                                        is_training=is_training,
                                                        global_pool=global_pool,
                                                        output_stride=output_stride,
                                                        spatial_squeeze=spatial_squeeze
                                                        )

        return logits, end_points
    if net_name == 'resnet_v1_101':
        FLAGS = get_flags_byname(net_name)
        with slim.arg_scope(resnet_v1.resnet_arg_scope(weight_decay=FLAGS.weight_decay)):
            logits, end_points = resnet_v1.resnet_v1_101(inputs=inputs,
                                                         num_classes=num_classes,
                                                         is_training=is_training,
                                                         global_pool=global_pool,
                                                         output_stride=output_stride,
                                                         spatial_squeeze=spatial_squeeze
                                                         )
        return logits, end_points
    if net_name == 'pvanet':
        FLAGS = get_flags_byname(net_name)
        with slim.arg_scope(pvanet.pvanet_scope(is_training=is_training,
                 weights_initializer = slim.xavier_initializer(),
                 batch_norm_param_initializer = None,
                 beta_initializer = tf.zeros_initializer(),
                 gamma_initializer = tf.ones_initializer(),
                 weight_decay = 0.99)):
            logits, end_points = pvanet.pvanet(net=inputs,
                                               include_last_bn_relu = True)
        return logits, end_points

    if net_name == 'vgg_16':
        FLAGS = get_flags_byname(net_name)
        with slim.arg_scope(vgg.vgg_arg_scope(weight_decay=FLAGS.weight_decay)):
            logits, end_points = vgg.vgg_16(inputs=inputs,
                                            num_classes=num_classes,
                                            is_training=is_training,
                                            dropout_keep_prob=0.5,
                                            spatial_squeeze=spatial_squeeze,
                                            )
        return logits, end_points
    # if net_name == 'inception_resnet_v2':
    #     FLAGS = get_flags_byname(net_name)
    #     with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope(weight_decay=FLAGS.weight_decay)):
    #         logits, end_points = inception_resnet_v2.inception_resnet_v2(inputs=inputs,
    #                                         num_classes=num_classes,
    #                                         is_training=is_training,
    #                                         dropout_keep_prob=0.8,
    #                                         )
    #     return logits, end_points
    if net_name == 'inception_resnet':
        FLAGS = get_flags_byname(net_name)
        arg_sc = inception_resnet_v2.inception_resnet_v2_arg_scope(weight_decay=FLAGS.weight_decay)
        with slim.arg_scope(arg_sc):
            logits, end_points = inception_resnet_v2.inception_resnet_v2(inputs=inputs,
                                                                         num_classes=num_classes,
                                                                         is_training=is_training)

        return logits, end_points

    if net_name == 'inception_v4':
        FLAGS = get_flags_byname(net_name)
        arg_sc = inception_v4.inception_v4_arg_scope(weight_decay=FLAGS.weight_decay)
        with slim.arg_scope(arg_sc):
            logits, end_points = inception_v4.inception_v4(inputs=inputs,
                                                           num_classes=num_classes,
                                                           is_training=is_training)
        return logits, end_points

    if net_name == 'mobilenet_224':
        FLAGS = get_flags_byname(net_name)
        with slim.arg_scope(mobilenet_v1.mobilenet_v1_arg_scope(weight_decay=FLAGS.weight_decay)):
            logits, end_points = mobilenet_v1.mobilenet_v1(inputs=inputs,
                                                         num_classes=num_classes,
                                                         is_training=is_training,
                                                         spatial_squeeze=spatial_squeeze
                                                         )
        return logits, end_points
    #
    # if net_name == 'vgg16':
    #     FLAGS = get_flags_byname(net_name)
    #     arg_sc = vgg.vgg_arg_scope(weight_decay=FLAGS.weight_decay)
    #     with slim.arg_scope(arg_sc):
    #         logits, end_points = vgg.vgg_16(inputs=inputs,
    #                                         num_classes=num_classes,
    #                                         is_training=is_training)
    #     return logits, end_points
