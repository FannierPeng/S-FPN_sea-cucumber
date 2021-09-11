# # -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(1234)
import tensorflow as tf
import tensorflow.contrib.slim as slim
from libs.configs import cfgs
from libs.box_utils import make_anchor, nms
from libs.box_utils import boxes_utils
from libs.box_utils import iou
from libs.box_utils import encode_and_decode
from libs.box_utils.show_box_in_tensor import draw_box_with_color
from libs.losses import losses

DEBUG = True


class RPN(object):
    def __init__(self, net_name, inputs, gtboxes_and_label,
                 is_training,
                 share_net,
                 anchor_ratios,
                 anchor_scales,
                 scale_factors,
                 base_anchor_size_list,  # P2, P3, P4, P5, P6
                 stride,
                 level,
                 top_k_nms,
                 share_head=False,
                 rpn_nms_iou_threshold=0.7,
                 max_proposals_num=300,
                 rpn_iou_positive_threshold=0.7,
                 rpn_iou_negative_threshold=0.3,  # iou>=0.7 is positive box, iou< 0.3 is negative
                 rpn_mini_batch_size=256,
                 rpn_positives_ratio=0.5,
                 remove_outside_anchors=False,  # whether remove anchors outside
                 rpn_weight_decay=0.0001,
                 ):

        self.net_name = net_name
        self.img_batch = inputs
        self.gtboxes_and_label = gtboxes_and_label  # shape is [M. 5],

        self.base_anchor_size_list = base_anchor_size_list

        self.anchor_ratios = tf.constant(anchor_ratios, dtype=tf.float32)
        self.anchor_scales = tf.constant(anchor_scales, dtype=tf.float32)
        self.share_head = share_head
        self.num_of_anchors_per_location = len(anchor_scales) * len(anchor_ratios)

        self.scale_factors = scale_factors
        self.stride = stride
        self.level = level
        self.top_k_nms = top_k_nms

        self.rpn_nms_iou_threshold = rpn_nms_iou_threshold
        self.max_proposals_num = max_proposals_num

        self.rpn_iou_positive_threshold = rpn_iou_positive_threshold
        self.rpn_iou_negative_threshold = rpn_iou_negative_threshold
        self.rpn_mini_batch_size = rpn_mini_batch_size
        self.rpn_positives_ratio = rpn_positives_ratio
        self.remove_outside_anchors = remove_outside_anchors
        self.rpn_weight_decay = rpn_weight_decay
        self.is_training = is_training
        self.share_net = share_net

        self.feature_maps_dict = self.get_feature_maps()
        self.feature_pyramid = self.build_feature_pyramid()

        self.anchors, self.rpn_encode_boxes, self.rpn_scores = self.get_anchors_and_rpn_predict()

    def get_feature_maps(self):

        '''
            Compared to https://github.com/KaimingHe/deep-residual-networks, the implementation of resnet_50 in slim
            subsample the output activations in the last residual unit of each block,
            instead of subsampling the input activations in the first residual unit of each block.
            The two implementations give identical results but the implementation of slim is more memory efficient.

            SO, when we build feature_pyramid, we should modify the value of 'C_*' to get correct spatial size feature maps.
            :return: feature maps
        '''

        with tf.variable_scope('get_feature_maps'):
            if self.net_name == 'resnet_v1_50':
                feature_maps_dict = {
                    'C2': self.share_net['resnet_v1_50/block1/unit_2/bottleneck_v1'],  # [56, 56]
                    'C3': self.share_net['resnet_v1_50/block2/unit_3/bottleneck_v1'],  # [28, 28]
                    'C4': self.share_net['resnet_v1_50/block3/unit_5/bottleneck_v1'],  # [14, 14]
                    'C5': self.share_net['resnet_v1_50/block4']  # [7, 7]
                }
            elif self.net_name == 'resnet_v1_101':
                feature_maps_dict = {
                    'C2': self.share_net['resnet_v1_101/block1/unit_2/bottleneck_v1'],  # [56, 56]
                    'C3': self.share_net['resnet_v1_101/block2/unit_3/bottleneck_v1'],  # [28, 28]
                    'C4': self.share_net['resnet_v1_101/block3/unit_22/bottleneck_v1'],  # [14, 14]
                    'C5': self.share_net['resnet_v1_101/block4']  # [7, 7]
                }
            elif self.net_name == 'pvanet':
                feature_maps_dict = {
                    'C2': self.share_net['conv2'],  # [128, 128]
                    'C3': self.share_net['conv3'],  # [64, 64]
                    'C4': self.share_net['conv4'],  # [32, 32]
                    'C5': self.share_net['conv5']  # [16, 16]
                }
            elif self.net_name == 'vgg_16':
                feature_maps_dict = {
                    'C2': self.share_net['vgg_16/conv2/conv2_2'],  # [128, 128]
                    'C3': self.share_net['vgg_16/conv3/conv3_3'],  # [64, 64]
                    'C4': self.share_net['vgg_16/conv4/conv4_3'],  # [32, 32]
                    'C5': self.share_net['vgg_16/conv5/conv5_3']  # [16, 16]
                }
            elif self.net_name == 'inception_resnet':
                feature_maps_dict = {
                    'C2': self.share_net['InceptionResnetV2/Conv2d_4a_3x3'],  # [71, 71]
                    'C3': self.share_net['InceptionResnetV2/Repeat/block35_10'],  # [35, 35]
                    'C4': self.share_net['InceptionResnetV2/Repeat_1/block17_20'],  # [17, 17]
                    'C5': self.share_net['InceptionResnetV2/Repeat_2/block8_9']  # [8, 8]
            }
            elif self.net_name == 'inception_v4':
                feature_maps_dict = {
                'C2': self.share_net['InceptionV4/Mixed_4a'],  # [71, 71]
                'C3': self.share_net['InceptionV4/Mixed_5e'],  # [35, 35]
                'C4': self.share_net['InceptionV4/Mixed_6h'],  # [17, 17]
                'C5': self.share_net['InceptionV4/Mixed_7d']  # [8, 8]
            }
            elif self.net_name == 'mobilenet_224':
                feature_maps_dict = {
                'C2': self.share_net['MobilenetV1/Conv2d_4_depthwise'],  # [56, 56]
                'C3': self.share_net['MobilenetV1/Conv2d_6_depthwise'],  # [28, 28]
                'C4': self.share_net['MobilenetV1/Conv2d_12_depthwise'],  # [14, 14]
                'C5': self.share_net['MobilenetV1/Conv2d_13_depthwise']  # [7, 7]
            }
            else:
                raise Exception('get no feature maps')

            return feature_maps_dict

    def build_feature_pyramid(self):

        '''
        reference: https://github.com/CharlesShang/FastMaskRCNN
        build P2, P3, P4, P5
        :return: multi-scale feature map
        '''

        feature_pyramid = {}
        with tf.variable_scope('build_feature_pyramid'):
            with slim.arg_scope([slim.conv2d], weights_regularizer=slim.l2_regularizer(self.rpn_weight_decay)):
                feature_pyramid['P5'] = slim.conv2d(self.feature_maps_dict['C5'],
                                                    num_outputs=256,
                                                    kernel_size=[1, 1],
                                                    stride=1,
                                                    scope='build_P5')

                feature_pyramid['P6'] = slim.max_pool2d(feature_pyramid['P5'],
                                                        kernel_size=[2, 2], stride=2, scope='build_P6')
                # P6 is down sample of P5

                for layer in range(4, 1, -1):
                    p, c = feature_pyramid['P' + str(layer + 1)], self.feature_maps_dict['C' + str(layer)]
                    up_sample_shape = tf.shape(c)
                    up_sample = tf.image.resize_nearest_neighbor(p, [up_sample_shape[1], up_sample_shape[2]],
                                                                 name='build_P%d/up_sample_nearest_neighbor' % layer)

                    c = slim.conv2d(c, num_outputs=256, kernel_size=[1, 1], stride=1,
                                    scope='build_P%d/reduce_dimension' % layer)
                    p = up_sample + c
                    p = slim.conv2d(p, 256, kernel_size=[3, 3], stride=1,
                                    padding='SAME', scope='build_P%d/avoid_aliasing' % layer)


                    feature_pyramid['P' + str(layer)] = p
                ## 一阶short cut
                # up_sample_shape = tf.shape(feature_pyramid['P2'])
                # up_sample8x = tf.image.resize_nearest_neighbor(feature_pyramid['P5'], [up_sample_shape[1], up_sample_shape[2]],
                #                                                  name='build_P2res/up_sample_nearest_neighbor')
                # n_in = feature_pyramid['P5'].get_shape()[-1]
                # if n_in != 256:  # projection
                #     shortcut = tf.layers.conv2d(up_sample8x, 256, 1, 1, name='projection_res')
                # else:
                #     shortcut = up_sample8x  # identical mapping
                # p2=feature_pyramid['P2']
                # if self.is_training is True:
                #     # 训练模式 使用指数加权函数不断更新均值和方差
                #     P2 = tf.contrib.layers.batch_norm(inputs=p2, decay=0.9, updates_collections=None,
                #                                         is_training=True)
                # else:
                #     # 测试模式 不更新均值和方差，直接使用
                #     P2 = tf.contrib.layers.batch_norm(inputs=p2, decay=0.9, updates_collections=None,
                #                                         is_training=False)
                # P2 = tf.nn.relu(P2)
                # feature_pyramid['P2'] = P2 + shortcut

                ## 三阶short cut
                up_sample_shape1 = tf.shape(feature_pyramid['P4'])
                up_sample2x_1 = tf.image.resize_nearest_neighbor(feature_pyramid['P5'],
                                                               [up_sample_shape1[1], up_sample_shape1[2]],
                                                               name='build_P4res/up_sample_nearest_neighbor')
                n_in = feature_pyramid['P5'].get_shape()[-1]
                if n_in != 256:  # projection
                    shortcut1 = tf.layers.conv2d(up_sample2x_1, 256, 1, 1, name='projection_res1')
                else:
                    shortcut1 = up_sample2x_1  # identical mapping
                p4 = feature_pyramid['P4']
                if self.is_training is True:
                    # 训练模式 使用指数加权函数不断更新均值和方差
                    P4 = tf.contrib.layers.batch_norm(inputs=p4,decay=0.9, updates_collections=None,
                                                        is_training=True)
                else:
                    # 测试模式 不更新均值和方差，直接使用
                    P4 = tf.contrib.layers.batch_norm(inputs=p4, decay=0.9, updates_collections=None,
                                                        is_training=False)
                P4 = tf.nn.elu(P4)
                feature_pyramid['P4'] = P4 + shortcut1

                up_sample_shape2 = tf.shape(feature_pyramid['P3'])
                up_sample2x_2 = tf.image.resize_nearest_neighbor(feature_pyramid['P4'],
                                                               [up_sample_shape2[1], up_sample_shape2[2]],
                                                               name='build_P3res/up_sample_nearest_neighbor')
                n_in = feature_pyramid['P4'].get_shape()[-1]
                if n_in != 256:  # projection
                    shortcut2 = tf.layers.conv2d(up_sample2x_2, 256, 1, 1, name='projection_res2')
                else:
                    shortcut2 = up_sample2x_2  # identical mapping
                p3 = feature_pyramid['P3']
                if self.is_training is True:
                    # 训练模式 使用指数加权函数不断更新均值和方差
                    P3 = tf.contrib.layers.batch_norm(inputs=p3, decay=0.9, updates_collections=None,
                                                      is_training=True)
                else:
                    # 测试模式 不更新均值和方差，直接使用
                    P3 = tf.contrib.layers.batch_norm(inputs=p3, decay=0.9, updates_collections=None,
                                                      is_training=False)
                P3 = tf.nn.elu(P3)
                feature_pyramid['P3'] = P3 + shortcut2


                up_sample_shape3 = tf.shape(feature_pyramid['P2'])
                up_sample2x_3 = tf.image.resize_nearest_neighbor(feature_pyramid['P3'],
                                                               [up_sample_shape3[1], up_sample_shape3[2]],
                                                               name='build_P2res/up_sample_nearest_neighbor')
                n_in = feature_pyramid['P3'].get_shape()[-1]
                if n_in != 256:  # projection
                    shortcut3 = tf.layers.conv2d(up_sample2x_3, 256, 1, 1, name='projection_res3')
                else:
                    shortcut3 = up_sample2x_3  # identical mapping
                p2 = feature_pyramid['P2']
                if self.is_training is True:
                    # 训练模式 使用指数加权函数不断更新均值和方差
                    P2 = tf.contrib.layers.batch_norm(inputs=p2, decay=0.9, updates_collections=None,
                                                      is_training=True)
                else:
                    # 测试模式 不更新均值和方差，直接使用
                    P2 = tf.contrib.layers.batch_norm(inputs=p2, decay=0.9, updates_collections=None,
                                                      is_training=False)
                P2 = tf.nn.elu(P2)
                feature_pyramid['P2'] = P2 + shortcut3

                # # 两阶short cut
                # up_sample_shape1 = tf.shape(feature_pyramid['P3'])
                # up_sample4x_1 = tf.image.resize_nearest_neighbor(feature_pyramid['P5'],
                #                                                [up_sample_shape1[1], up_sample_shape1[2]],
                #                                                name='build_P3res/up_sample_nearest_neighbor')
                # n_in = feature_pyramid['P5'].get_shape()[-1]
                # if n_in != 256:  # projection
                #     shortcut1 = tf.layers.conv2d(up_sample4x_1, 256, 1, 1, name='projection_res1')
                # else:
                #     shortcut1 = up_sample4x_1  # identical mapping
                # p3 = feature_pyramid['P3']
                # if self.is_training is True:
                #     # 训练模式 使用指数加权函数不断更新均值和方差
                #     P3 = tf.contrib.layers.batch_norm(inputs=p3,decay=0.9, updates_collections=None,
                #                                         is_training=True)
                # else:
                #     # 测试模式 不更新均值和方差，直接使用
                #     P3= tf.contrib.layers.batch_norm(inputs=p3, decay=0.9, updates_collections=None,
                #                                                         is_training=False)
                # P3 = tf.nn.relu(P3)
                # feature_pyramid['P3'] = P3 + shortcut1
                #
                #
                # up_sample_shape2 = tf.shape(feature_pyramid['P2'])
                # up_sample4x_2 = tf.image.resize_nearest_neighbor(feature_pyramid['P4'],
                #                                                [up_sample_shape2[1], up_sample_shape2[2]],
                #                                                name='build_P2res/up_sample_nearest_neighbor')
                #
                # n_in2 = feature_pyramid['P4'].get_shape()[-1]
                # if n_in2 != 256:  # projection
                #     shortcut2 = tf.layers.conv2d(up_sample4x_2, 256, 1, 1, name='projection_res2')
                # else:
                #     shortcut2 = up_sample4x_2  # identical mapping
                #
                # p2 = feature_pyramid['P2']
                # if self.is_training is True:
                #     # 训练模式 使用指数加权函数不断更新均值和方差
                #     P2 = tf.contrib.layers.batch_norm(inputs=p2, decay=0.9, updates_collections=None,
                #                                       is_training=True)
                # else:
                #     # 测试模式 不更新均值和方差，直接使用
                #     P2 = tf.contrib.layers.batch_norm(inputs=p2, decay=0.9, updates_collections=None,
                #                                       is_training=False)
                # P2 = tf.nn.relu(P2)
                # feature_pyramid['P2'] = P2 + shortcut2


                # # 五阶short cut
                # up_sample_shape1 = tf.shape(feature_pyramid['P4'])
                # up_sample2x_1 = tf.image.resize_nearest_neighbor(feature_pyramid['P5'],
                #                                                [up_sample_shape1[1], up_sample_shape1[2]],
                #                                                name='build_P4res/up_sample_nearest_neighbor')
                # n_in = feature_pyramid['P5'].get_shape()[-1]
                # if n_in != 256:  # projection
                #     shortcut1 = tf.layers.conv2d(up_sample2x_1, 256, 1, 1, name='projection_res1')
                # else:
                #     shortcut1 = up_sample2x_1  # identical mapping
                # p4 = feature_pyramid['P4']
                # if self.is_training is True:
                #     # 训练模式 使用指数加权函数不断更新均值和方差
                #     P4 = tf.contrib.layers.batch_norm(inputs=p4,decay=0.9, updates_collections=None,
                #                                         is_training=True)
                # else:
                #     # 测试模式 不更新均值和方差，直接使用
                #     P4 = tf.contrib.layers.batch_norm(inputs=p4, decay=0.9, updates_collections=None,
                #                                                         is_training=False)
                # P4 = tf.nn.relu(P4)
                # feature_pyramid['P4'] = P4 + shortcut1
                #
                #
                # up_sample_shape2 = tf.shape(feature_pyramid['P3'])
                # up_sample2x_2 = tf.image.resize_nearest_neighbor(feature_pyramid['P4'],
                #                                                [up_sample_shape2[1], up_sample_shape2[2]],
                #                                                name='build_P3_1res/up_sample_nearest_neighbor')
                # up_sample4x_1 = tf.image.resize_nearest_neighbor(feature_pyramid['P5'],
                #                                                [up_sample_shape2[1], up_sample_shape2[2]],
                #                                                name='build_P3_2res/up_sample_nearest_neighbor')
                # n_in1 = feature_pyramid['P4'].get_shape()[-1]
                # n_in2 = feature_pyramid['P5'].get_shape()[-1]
                # if n_in1 != 256:  # projection
                #     shortcut2 = tf.layers.conv2d(up_sample2x_2, 256, 1, 1, name='projection_res2')
                # else:
                #     shortcut2 = up_sample2x_2  # identical mapping
                # if n_in2 != 256:  # projection
                #     shortcut3 = tf.layers.conv2d(up_sample4x_1, 256, 1, 1, name='projection_res3')
                # else:
                #     shortcut3 = up_sample4x_1  # identical mapping
                # p3 = feature_pyramid['P3']
                # if self.is_training is True:
                #     # 训练模式 使用指数加权函数不断更新均值和方差
                #     P3 = tf.contrib.layers.batch_norm(inputs=p3, decay=0.9, updates_collections=None,
                #                                       is_training=True)
                # else:
                #     # 测试模式 不更新均值和方差，直接使用
                #     P3 = tf.contrib.layers.batch_norm(inputs=p3, decay=0.9, updates_collections=None,
                #                                       is_training=False)
                # P3 = tf.nn.relu(P3)
                # feature_pyramid['P3'] = P3 + shortcut2
                # feature_pyramid['P3'] = feature_pyramid['P3'] + shortcut3
                #
                #
                # up_sample_shape3 = tf.shape(feature_pyramid['P2'])
                # up_sample2x_3 = tf.image.resize_nearest_neighbor(feature_pyramid['P3'],
                #                                                [up_sample_shape3[1], up_sample_shape3[2]],
                #                                                name='build_P2_1res/up_sample_nearest_neighbor')
                # up_sample4x_2 = tf.image.resize_nearest_neighbor(feature_pyramid['P4'],
                #                                                  [up_sample_shape3[1], up_sample_shape3[2]],
                #                                                  name='build_P2_2res/up_sample_nearest_neighbor')
                # # up_sample8x = tf.image.resize_nearest_neighbor(feature_pyramid['P5'],
                # #                                                  [up_sample_shape3[1], up_sample_shape3[2]],
                # #                                                  name='build_P2_3res/up_sample_nearest_neighbor')
                # n_in1 = feature_pyramid['P3'].get_shape()[-1]
                # n_in2 = feature_pyramid['P4'].get_shape()[-1]
                # # n_in3 = feature_pyramid['P5'].get_shape()[-1]
                # if n_in1 != 256:  # projection
                #     shortcut4= tf.layers.conv2d(up_sample2x_3, 256, 1, 1, name='projection_res4')
                # else:
                #     shortcut4 = up_sample2x_3  # identical mapping
                # if n_in2 != 256:  # projection
                #     shortcut5 = tf.layers.conv2d(up_sample4x_2, 256, 1, 1, name='projection_res5')
                # else:
                #     shortcut5 = up_sample4x_2  # identical mapping
                # # if n_in3 != 256:  # projection
                # #     shortcut6= tf.layers.conv2d(up_sample8x, 256, 1, 1, name='projection_res6')
                # # else:
                # #     shortcut6 = up_sample8x  # identical mapping
                # p2 = feature_pyramid['P2']
                # if self.is_training is True:
                #     # 训练模式 使用指数加权函数不断更新均值和方差
                #     P2 = tf.contrib.layers.batch_norm(inputs=p2, decay=0.9, updates_collections=None,
                #                                       is_training=True)
                # else:
                #     # 测试模式 不更新均值和方差，直接使用
                #     P2 = tf.contrib.layers.batch_norm(inputs=p2, decay=0.9, updates_collections=None,
                #                                       is_training=False)
                # P2 = tf.nn.relu(P2)
                # feature_pyramid['P2'] = P2 + shortcut4
                # feature_pyramid['P2'] = feature_pyramid['P2'] + shortcut5
        return feature_pyramid

    def make_anchors(self):
        with tf.variable_scope('make_anchors'):
            anchor_list = []
            level_list = self.level
            with tf.name_scope('make_anchors_all_level'):
                for level, base_anchor_size, stride in zip(level_list, self.base_anchor_size_list, self.stride):
                    '''
                    (level, base_anchor_size) tuple:
                    (P2, 32), (P3, 64), (P4, 128), (P5, 256), (P6, 512)
                    (P2, 15), (P3, 25), (P4, 40), (P5, 60), (P6, 80)
                    STRIDE = [4, 8, 16, 32, 64]
                    '''
                    featuremap_height, featuremap_width = tf.shape(self.feature_pyramid[level])[1], \
                                                          tf.shape(self.feature_pyramid[level])[2]
                    # stride = base_anchor_size / 8.

                    # tmp_anchors = tf.py_func(
                    #     anchor_utils_pyfunc.make_anchors,
                    #     inp=[base_anchor_size, self.anchor_scales, self.anchor_ratios,
                    #          featuremap_height, featuremap_width, stride],
                    #     Tout=tf.float32
                    # )

                    tmp_anchors = make_anchor.make_anchors(base_anchor_size, self.anchor_scales, self.anchor_ratios,
                                                           featuremap_height,  featuremap_width, stride,
                                                           name='make_anchors_{}'.format(level))
                    tmp_anchors = tf.reshape(tmp_anchors, [-1, 4])
                    anchor_list.append(tmp_anchors)

                all_level_anchors = tf.concat(anchor_list, axis=0)
            return all_level_anchors

    def rpn_net(self):

        rpn_encode_boxes_list = []
        rpn_scores_list = []
        with tf.variable_scope('rpn_net'):
            with slim.arg_scope([slim.conv2d], weights_regularizer=slim.l2_regularizer(self.rpn_weight_decay)):
                for level in self.level:

                    if self.share_head:
                        reuse_flag = None if level == 'P2' else True
                        scope_list = ['conv2d_3x3', 'rpn_classifier', 'rpn_regressor']
                        # in the begining, we should create variables, then sharing variables in P3, P4, P5
                    else:
                        reuse_flag = None
                        scope_list = ['conv2d_3x3_'+level, 'rpn_classifier_'+level, 'rpn_regressor_'+level]

                    rpn_conv2d_3x3 = slim.conv2d(inputs=self.feature_pyramid[level],
                                                 num_outputs=512,
                                                 kernel_size=[3, 3],
                                                 stride=1,
                                                 scope=scope_list[0],
                                                 reuse=reuse_flag)
                    #short cut
                    # n_in = self.feature_pyramid[level].get_shape()[-1]
                    # if n_in != 512:  # projection
                    #     shortcut = tf.layers.conv2d(self.feature_pyramid[level], 512, 1, 1, name='projection'+level)
                    # else:
                    #     shortcut = self.feature_pyramid[level]  # identical mapping
                    # rpn_conv2d_res = tf.nn.relu(shortcut + rpn_conv2d_3x3)

                    rpn_box_scores = slim.conv2d(rpn_conv2d_3x3,
                                                 num_outputs=2 * self.num_of_anchors_per_location,
                                                 kernel_size=[1, 1],
                                                 stride=1,
                                                 scope=scope_list[1],
                                                 activation_fn=None,
                                                 reuse=reuse_flag)
                    rpn_encode_boxes = slim.conv2d(rpn_conv2d_3x3,
                                                   num_outputs=4 * self.num_of_anchors_per_location,
                                                   kernel_size=[1, 1],
                                                   stride=1,
                                                   scope=scope_list[2],
                                                   activation_fn=None,
                                                   reuse=reuse_flag)

                    rpn_box_scores = tf.reshape(rpn_box_scores, [-1, 2])
                    rpn_encode_boxes = tf.reshape(rpn_encode_boxes, [-1, 4])

                    rpn_scores_list.append(rpn_box_scores)
                    rpn_encode_boxes_list.append(rpn_encode_boxes)

                rpn_all_encode_boxes = tf.concat(rpn_encode_boxes_list, axis=0)
                rpn_all_boxes_scores = tf.concat(rpn_scores_list, axis=0)

            return rpn_all_encode_boxes, rpn_all_boxes_scores

    def get_anchors_and_rpn_predict(self):

        anchors = self.make_anchors()
        rpn_encode_boxes, rpn_scores = self.rpn_net()

        with tf.name_scope('get_anchors_and_rpn_predict'):
            if self.is_training:
                if self.remove_outside_anchors:
                    valid_indices = boxes_utils.filter_outside_boxes(boxes=anchors,
                                                                     img_h=tf.shape(self.img_batch)[1],
                                                                     img_w=tf.shape(self.img_batch)[2])
                    valid_anchors = tf.gather(anchors, valid_indices)
                    rpn_valid_encode_boxes = tf.gather(rpn_encode_boxes, valid_indices)
                    rpn_valid_scores = tf.gather(rpn_scores, valid_indices)

                    return valid_anchors, rpn_valid_encode_boxes, rpn_valid_scores
                else:
                    return anchors, rpn_encode_boxes, rpn_scores
            else:
                return anchors, rpn_encode_boxes, rpn_scores

    def rpn_find_positive_negative_samples(self, anchors):
        '''
        assign anchors targets: object or background.
        :param anchors: [valid_num_of_anchors, 4]. use N to represent valid_num_of_anchors

        :return:labels. anchors_matched_gtboxes, object_mask

        labels shape is [N, ].  positive is 1, negative is 0, ignored is -1
        anchor_matched_gtboxes. each anchor's gtbox(only positive box has gtbox)shape is [N, 4]
        object_mask. tf.float32. 1.0 represent box is object, 0.0 is others. shape is [N, ]
        '''
        with tf.variable_scope('rpn_find_positive_negative_samples'):
            gtboxes = tf.reshape(self.gtboxes_and_label[:, :-1], [-1, 4])
            gtboxes = tf.cast(gtboxes, tf.float32)

            ious = iou.iou_calculate(anchors, gtboxes)  # [N, M]

            max_iou_each_row = tf.reduce_max(ious, axis=1)

            labels = tf.ones(shape=[tf.shape(anchors)[0], ], dtype=tf.float32) * (-1)  # [N, ] # ignored is -1

            matchs = tf.cast(tf.argmax(ious, axis=1), tf.int32)

            # an anchor that has an IoU overlap higher than 0.7 with any ground-truth box
            positives1 = tf.greater_equal(max_iou_each_row, self.rpn_iou_positive_threshold)  # iou >= 0.7 is positive

            # to avoid none of boxes iou >= 0.7, use max iou boxes as positive
            max_iou_each_column = tf.reduce_max(ious, 0)
            # the anchor/anchors with the highest Intersection-over-Union (IoU) overlap with a ground-truth box
            positives2 = tf.reduce_sum(tf.cast(tf.equal(ious, max_iou_each_column), tf.float32), axis=1)

            positives = tf.logical_or(positives1, tf.cast(positives2, tf.bool))

            labels += 2 * tf.cast(positives, tf.float32)  # Now, positive is 1, ignored and background is -1

            # object_mask = tf.cast(positives, tf.float32)  # 1.0 is object, 0.0 is others

            # matchs = matchs * tf.cast(positives, dtype=matchs.dtype)  # remove background and ignored
            anchors_matched_gtboxes = tf.gather(gtboxes, matchs)  # [N, 4]
            # background's gtboxes tmp set the first gtbox, it dose not matter, because use object_mask will ignored it

            negatives = tf.less(max_iou_each_row, self.rpn_iou_negative_threshold)
            negatives = tf.logical_and(negatives, tf.greater_equal(max_iou_each_row, 0.1))

            labels = labels + tf.cast(negatives, tf.float32)  # [N, ] positive is >=1.0, negative is 0, ignored is -1.0
            '''
            Need to note: when opsitive, labels may >= 1.0.
            Because, when all the iou< 0.7, we set anchors having max iou each column as positive.
            these anchors may have iou < 0.3.
            In the begining, labels is [-1, -1, -1...-1]
            then anchors having iou<0.3 as well as are max iou each column will be +1.0.
            when decide negatives, because of iou<0.3, they add 1.0 again.
            So, the final result will be 2.0

            So, when opsitive, labels may in [1.0, 2.0]. that is labels >=1.0
            '''
            positives = tf.cast(tf.greater_equal(labels, 1.0), tf.float32)
            ignored = tf.cast(tf.equal(labels, -1.0), tf.float32) * -1

            labels = positives + ignored
            object_mask = tf.cast(positives, tf.float32)  # 1.0 is object, 0.0 is others

            return labels, anchors_matched_gtboxes, object_mask

    def make_minibatch(self, valid_anchors):
        with tf.variable_scope('rpn_minibatch'):

            # in labels(shape is [N, ]): 1 is positive, 0 is negative, -1 is ignored
            labels, anchor_matched_gtboxes, object_mask = \
                self.rpn_find_positive_negative_samples(valid_anchors)  # [num_of_valid_anchors, ]

            positive_indices = tf.reshape(tf.where(tf.equal(labels, 1.0)), [-1])  # use labels is same as object_mask

            num_of_positives = tf.minimum(tf.shape(positive_indices)[0],
                                          tf.cast(self.rpn_mini_batch_size * self.rpn_positives_ratio, tf.int32))

            # num of positives <= minibatch_size * 0.5
            positive_indices = tf.random_shuffle(positive_indices)
            positive_indices = tf.slice(positive_indices,
                                        begin=[0],
                                        size=[num_of_positives])

            negatives_indices = tf.reshape(tf.where(tf.equal(labels, 0.0)), [-1])
            num_of_negatives = tf.minimum(self.rpn_mini_batch_size - num_of_positives,
                                          tf.shape(negatives_indices)[0])

            negatives_indices = tf.random_shuffle(negatives_indices)
            negatives_indices = tf.slice(negatives_indices, begin=[0], size=[num_of_negatives])

            minibatch_indices = tf.concat([positive_indices, negatives_indices], axis=0)
            minibatch_indices = tf.random_shuffle(minibatch_indices)

            minibatch_anchor_matched_gtboxes = tf.gather(anchor_matched_gtboxes, minibatch_indices)
            object_mask = tf.gather(object_mask, minibatch_indices)
            labels = tf.cast(tf.gather(labels, minibatch_indices), tf.int32)
            labels_one_hot = tf.one_hot(labels, depth=2)
            return minibatch_indices, minibatch_anchor_matched_gtboxes, object_mask, labels_one_hot

    def rpn_losses(self):
        with tf.variable_scope('rpn_losses'):
            minibatch_indices, minibatch_anchor_matched_gtboxes, object_mask, minibatch_labels_one_hot = \
                self.make_minibatch(self.anchors)

            minibatch_anchors = tf.gather(self.anchors, minibatch_indices)
            minibatch_encode_boxes = tf.gather(self.rpn_encode_boxes, minibatch_indices)
            minibatch_boxes_scores = tf.gather(self.rpn_scores, minibatch_indices)

            # encode gtboxes
            minibatch_encode_gtboxes = encode_and_decode.encode_boxes(unencode_boxes=minibatch_anchor_matched_gtboxes,
                                                                      reference_boxes=minibatch_anchors,
                                                                      scale_factors=self.scale_factors)

            positive_anchors_in_img = draw_box_with_color(self.img_batch,
                                                          minibatch_anchors * tf.expand_dims(object_mask, 1),
                                                          text=tf.shape(tf.where(tf.equal(object_mask, 1.0)))[0])

            negative_mask = tf.cast(tf.logical_not(tf.cast(object_mask, tf.bool)), tf.float32)
            negative_anchors_in_img = draw_box_with_color(self.img_batch,
                                                          minibatch_anchors * tf.expand_dims(negative_mask, 1),
                                                          text=tf.shape(tf.where(tf.equal(object_mask, 0.0)))[0])

            minibatch_decode_boxes = encode_and_decode.decode_boxes(encode_boxes=minibatch_encode_boxes,
                                                                    reference_boxes=minibatch_anchors,
                                                                    scale_factors=self.scale_factors)

            tf.summary.image('/positive_anchors', positive_anchors_in_img)
            tf.summary.image('/negative_anchors', negative_anchors_in_img)
            top_k_scores, top_k_indices = tf.nn.top_k(minibatch_boxes_scores[:, 1], k=1)

            top_detections_in_img = draw_box_with_color(self.img_batch,
                                                        tf.gather(minibatch_decode_boxes, top_k_indices),
                                                        text=tf.shape(top_k_scores)[0])
            tf.summary.image('/top_1', top_detections_in_img)

            # losses
            with tf.variable_scope('rpn_location_loss'):
                location_loss = losses.l1_smooth_losses(predict_boxes=minibatch_encode_boxes,
                                                        gtboxes=minibatch_encode_gtboxes,
                                                        object_weights=object_mask)
                slim.losses.add_loss(location_loss)  # add smooth l1 loss to losses collection

            with tf.variable_scope('rpn_classification_loss'):


                # logits = tf.cast(minibatch_boxes_scores, tf.float32)
                # onehot_labels = tf.cast(minibatch_labels_one_hot, tf.float32)
                # one = tf.ones(shape=tf.shape(onehot_labels), dtype=tf.float32)
                # predictions_pt = tf.where(tf.equal(onehot_labels, 1), logits, 1-logits)
                #
                # # add small value to avoid
                # alpha_t = tf.scalar_mul(0.25, one)
                # alpha_t = tf.where(tf.equal(onehot_labels, 1), alpha_t, 1 - alpha_t)
                # gamma = tf.scalar_mul(2, one)
                # new_gamma = tf.where(tf.less(predictions_pt, 0.5), -gamma, gamma)
                # classification_loss = tf.multiply(tf.multiply(alpha_t, slim.losses.softmax_cross_entropy(logits=logits,
                #                                                   onehot_labels=onehot_labels)), tf.pow((1-predictions_pt), 2))
                # # classification_loss = tf.multiply(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                # #                                                    labels=onehot_labels), tf.pow((1-predictions_pt), 2))
                # classification_loss = tf.reduce_sum(classification_loss[:,0]+classification_loss[:,1])
                # # classification_loss = slim.losses.softmax_cross_entropy(logits=tf.clip_by_value(minibatch_boxes_scores,1e-8,tf.reduce_max(minibatch_boxes_scores)),
                # #                                                         onehot_labels=minibatch_labels_one_hot)
                classification_loss = slim.losses.softmax_cross_entropy(
                    logits=minibatch_boxes_scores,
                    onehot_labels=minibatch_labels_one_hot)
            return location_loss, classification_loss

    def rpn_proposals(self):
        with tf.variable_scope('rpn_proposals'):
            rpn_decode_boxes = encode_and_decode.decode_boxes(encode_boxes=self.rpn_encode_boxes,
                                                              reference_boxes=self.anchors,
                                                              scale_factors=self.scale_factors)

            if not self.is_training:  # when test, clip proposals to img boundaries
                img_shape = tf.shape(self.img_batch)
                rpn_decode_boxes = boxes_utils.clip_boxes_to_img_boundaries(rpn_decode_boxes, img_shape)

            rpn_softmax_scores = slim.softmax(self.rpn_scores)
            rpn_object_score = rpn_softmax_scores[:, 1]  # second column represent object

            if self.top_k_nms:
                rpn_object_score, top_k_indices = tf.nn.top_k(rpn_object_score, k=self.top_k_nms)
                rpn_decode_boxes = tf.gather(rpn_decode_boxes, top_k_indices)

            valid_indices = nms.non_maximal_suppression(boxes=rpn_decode_boxes,
                                                        scores=rpn_object_score,
                                                        max_output_size=self.max_proposals_num,
                                                        iou_threshold=self.rpn_nms_iou_threshold)

            valid_boxes = tf.gather(rpn_decode_boxes, valid_indices)
            valid_scores = tf.gather(rpn_object_score, valid_indices)
            rpn_proposals_boxes, rpn_proposals_scores = tf.cond(
                tf.less(tf.shape(valid_boxes)[0], self.max_proposals_num),
                lambda: boxes_utils.padd_boxes_with_zeros(valid_boxes, valid_scores,
                                                          self.max_proposals_num),
                lambda: (valid_boxes, valid_scores))

            return rpn_proposals_boxes, rpn_proposals_scores
