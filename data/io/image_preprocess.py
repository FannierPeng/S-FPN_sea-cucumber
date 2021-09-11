# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import numpy as np
np.random.seed(1)
from tensorflow import set_random_seed
set_random_seed(1234)
import tensorflow as tf
# import tensorlayer as tl

# import


def short_side_resize(img_tensor, gtboxes_and_label, target_shortside_len):
    '''

    :param img_tensor:[h, w, c], gtboxes_and_label:[-1, 5]
    :param target_shortside_len:
    :return:
    '''

    h, w = tf.shape(img_tensor)[0], tf.shape(img_tensor)[1]

    new_h, new_w = tf.cond(tf.less(h, w),
                           true_fn=lambda: (target_shortside_len, target_shortside_len * w//h),
                           false_fn=lambda: (target_shortside_len * h//w,  target_shortside_len))

    img_tensor = tf.expand_dims(img_tensor, axis=0)
    img_tensor = tf.image.resize_bilinear(img_tensor, [new_h, new_w])

    ymin, xmin, ymax, xmax, label = tf.unstack(gtboxes_and_label, axis=1)

    xmin, xmax = xmin * new_w//w, xmax * new_w//w
    ymin, ymax = ymin * new_h//h, ymax * new_h//h

    img_tensor = tf.squeeze(img_tensor, axis=0) # ensure imgtensor rank is 3
    return img_tensor, tf.transpose(tf.stack([ymin, xmin, ymax, xmax, label], axis=0))


def short_side_resize_for_inference_data(img_tensor, target_shortside_len, is_resize=True):
    h, w, = tf.shape(img_tensor)[0], tf.shape(img_tensor)[1]

    img_tensor = tf.expand_dims(img_tensor, axis=0)
    if is_resize:
        new_h, new_w = tf.cond(tf.less(h, w),
                               true_fn=lambda: (target_shortside_len, target_shortside_len * w // h),
                               false_fn=lambda: (target_shortside_len * h // w, target_shortside_len))
        img_tensor = tf.image.resize_bilinear(img_tensor, [new_h, new_w])

    return img_tensor  # [1, h, w, c]

#增强之左右翻转
def flip_left_right(img_tensor, gtboxes_and_label):
    h, w = tf.shape(img_tensor)[0], tf.shape(img_tensor)[1]
    img_tensor = tf.image.flip_left_right(img_tensor)

    ymin, xmin, ymax, xmax, label = tf.unstack(gtboxes_and_label, axis=1)
    new_xmin = w - xmax
    new_xmax = w - xmin
    # return img_tensor, tf.transpose(tf.stack([new_xmin, ymin, new_xmax, ymax, label], axis=0))
    return img_tensor, tf.transpose(tf.stack([ymin, new_xmin, ymax, new_xmax, label], axis=0))


def random_flip_left_right(img_tensor, gtboxes_and_label):

    img_tensor, gtboxes_and_label = tf.cond(tf.less(tf.random_uniform(shape=[], minval=0, maxval=1), 0.5),
                                            lambda: flip_left_right(img_tensor, gtboxes_and_label),
                                            lambda: (img_tensor, gtboxes_and_label))

    return img_tensor,  gtboxes_and_label

#增强之上下翻转
def flip_up_down(img_tensor, gtboxes_and_label):
    h, w = tf.shape(img_tensor)[0], tf.shape(img_tensor)[1]
    img_tensor = tf.image.flip_up_down(img_tensor)

    ymin, xmin, ymax, xmax, label = tf.unstack(gtboxes_and_label, axis=1)
    new_ymin = h - ymax
    new_ymax = h - ymin
    return img_tensor, tf.transpose(tf.stack([new_ymin, xmin, new_ymax, xmax, label], axis=0))


def random_flip_up_down(img_tensor, gtboxes_and_label):

    img_tensor, gtboxes_and_label = tf.cond(tf.less(tf.random_uniform(shape=[], minval=0, maxval=1), 0.5),
                                            lambda: flip_up_down(img_tensor, gtboxes_and_label),
                                            lambda: (img_tensor, gtboxes_and_label))

    return img_tensor,  gtboxes_and_label
#
# #增强之对角线翻转
# def transpose_image(img_tensor, gtboxes_and_label):
#     h, w = tf.shape(img_tensor)[0], tf.shape(img_tensor)[1]
#     img_tensor = tf.image.transpose_image(img_tensor)
#
#     ymin, xmin, ymax, xmax, label = tf.unstack(gtboxes_and_label, axis=1)
#     new_ymin = xmin
#     new_ymax = xmax
#     new_xmin = ymin
#     new_xmax = ymax
#     return img_tensor, tf.transpose(tf.stack([new_ymin, new_xmin, new_ymax, new_xmax, label], axis=0))
#
#
# def random_transpose_image(img_tensor, gtboxes_and_label):
#
#     img_tensor, gtboxes_and_label = tf.cond(tf.less(tf.random_uniform(shape=[], minval=0, maxval=1), 1/3),
#                                             lambda: transpose_image(img_tensor, gtboxes_and_label),
#                                             lambda: (img_tensor, gtboxes_and_label))
#
#     return img_tensor,  gtboxes_and_label
#

#增强之加噪声
def add_noise(img_tensor, gtboxes_and_label):
    h, w = tf.shape(img_tensor)[0], tf.shape(img_tensor)[1]
    img_tensor = tf.add(img_tensor,tf.truncated_normal(tf.shape(img_tensor), stddev = 0.1, seed=1234))
    return img_tensor, gtboxes_and_label


def random_add_noise(img_tensor, gtboxes_and_label):

    img_tensor, gtboxes_and_label = tf.cond(tf.less(tf.random_uniform(shape=[], minval=0, maxval=1), 0.25),
                                            lambda: add_noise(img_tensor, gtboxes_and_label),
                                            lambda: (img_tensor, gtboxes_and_label))

    return img_tensor,  gtboxes_and_label

#增强之旋转90度
def rotate_90(img_tensor, gtboxes_and_label):
    h, w = tf.shape(img_tensor)[0], tf.shape(img_tensor)[1]
    img_tensor = tf.image.rot90(img_tensor,k=1)

    ymin, xmin, ymax, xmax, label = tf.unstack(gtboxes_and_label, axis=1)
    new_ymin = w - xmax
    new_ymax = w - xmin
    new_xmin = ymin
    new_xmax = ymax
    return img_tensor, tf.transpose(tf.stack([new_ymin, new_xmin, new_ymax, new_xmax, label], axis=0))


def random_rotate_90(img_tensor, gtboxes_and_label):

    img_tensor, gtboxes_and_label = tf.cond(tf.less(tf.random_uniform(shape=[], minval=0, maxval=1), 0.5),
                                            lambda: rotate_90(img_tensor, gtboxes_and_label),
                                            lambda: (img_tensor, gtboxes_and_label))

    return img_tensor,  gtboxes_and_label
# #增强之旋转180度
# def rotate_180(img_tensor, gtboxes_and_label):
#     h, w = tf.shape(img_tensor)[0], tf.shape(img_tensor)[1]
#     img_tensor = tf.image.transpose_image(img_tensor)
#
#     ymin, xmin, ymax, xmax, label = tf.unstack(gtboxes_and_label, axis=1)
#     new_ymin = h - ymax
#     new_ymax = h - ymin
#     new_xmin = w - xmax
#     new_xmax = w - xmin
#     return img_tensor, tf.transpose(tf.stack([new_ymin, new_xmin, new_ymax, new_xmax, label], axis=0))
#
#
# def random_rotate_180(img_tensor, gtboxes_and_label):
#
#     img_tensor, gtboxes_and_label = tf.cond(tf.less(tf.random_uniform(shape=[], minval=0, maxval=1), 0.25),
#                                             lambda: rotate_180(img_tensor, gtboxes_and_label),
#                                             lambda: (img_tensor, gtboxes_and_label))
#
#     return img_tensor,  gtboxes_and_label
#增强之旋转270度
def rotate_270(img_tensor, gtboxes_and_label):
    h, w = tf.shape(img_tensor)[0], tf.shape(img_tensor)[1]
    img_tensor = tf.image.rot90(img_tensor,k=1)

    ymin, xmin, ymax, xmax, label = tf.unstack(gtboxes_and_label, axis=1)
    new_ymin = xmin
    new_ymax = xmax
    new_xmin = h - ymax
    new_xmax = h - ymin
    return img_tensor, tf.transpose(tf.stack([new_ymin, new_xmin, new_ymax, new_xmax, label], axis=0))


def random_rotate_270(img_tensor, gtboxes_and_label):

    img_tensor, gtboxes_and_label = tf.cond(tf.less(tf.random_uniform(shape=[], minval=0, maxval=1), 1/3.),
                                            lambda: rotate_270(img_tensor, gtboxes_and_label),
                                            lambda: (img_tensor, gtboxes_and_label))

    return img_tensor,  gtboxes_and_label

#增强之调整
def adjust(img_tensor, gtboxes_and_label):
    h, w = tf.shape(img_tensor)[0], tf.shape(img_tensor)[1]
    img_tensor = tf.image.random_brightness(img_tensor, max_delta=0.5)
    img_tensor = tf.image.random_contrast(img_tensor, 0.1, 0.6)
    img_tensor = tf.image.random_hue(img_tensor, 0.5)
    img_tensor = tf.image.random_saturation(img_tensor, 0, 5)
    return img_tensor, gtboxes_and_label


def random_adjust(img_tensor, gtboxes_and_label):

    img_tensor, gtboxes_and_label = tf.cond(tf.less(tf.random_uniform(shape=[], minval=0, maxval=1), 0.09),
                                            lambda: adjust(img_tensor, gtboxes_and_label),
                                            lambda: (img_tensor, gtboxes_and_label))

    return img_tensor,  gtboxes_and_label




