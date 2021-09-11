#!/user/bin/env python
# _*_ coding:utf-8 _*_
# -*- coding:utf-8 -*-
# Author:Richard Fang
"""
This is a Python version used to implement the Soft NMS algorithm.
Original Paper：Improving Object Detection With One Line of Code
"""
import numpy as np
import tensorflow as tf
from keras import backend as K
def cond(keep, tBD, tscore, sigma, n):
    return tf.not_equal(tf.reduce_sum(tscore), 0)

# def cond1(i, n):
#     return i < n
# def body1(i, n):
#     add =
#     i =i + 1
#     return add
# def body2(keep, maxscore_index):
#     keep = keep + maxscore_index
#     return keep
# def body3(keep, n, maxscore_index):
#     i = 0
#     keep = keep + tf.concat(tf.while_loop(cond1, body1, [i, n]), maxscore_index, 0)
#     n = n + 1
#     return keep
def body(keep, tBD, tscore, sigma, n):
    y1 = tBD[:, 0]
    x1 = tBD[:, 1]
    y2 = tBD[:, 2]
    x2 = tBD[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # 从大到小排列，取index
    maxscore_index = tf.argmax(tscore, 0)
    maxscore = tf.reduce_max(tscore)
    M = tBD[maxscore_index, :]
    M_area = (M[3] - M[1] + 1) * (M[2] - M[0] + 1)
    # 计算窗口M与其他所有窗口的交叠部分的面积，矩阵计算
    xx1 = tf.maximum(x1, M[1])
    yy1 = tf.maximum(y1, M[0])
    xx2 = tf.minimum(x2, M[3])
    yy2 = tf.minimum(y2, M[2])

    w = tf.maximum(0.0, xx2 - xx1 + 1)
    h = tf.maximum(0.0, yy2 - yy1 + 1)
    inter = w * h
    # 交/并得到iou值
    ovr = inter / (areas + M_area - inter)
    zero2 = tf.zeros(shape=tf.shape(tscore), dtype=tf.float32)
    tscore = tf.cast(tscore, dtype=tf.float32)
    tscore = tf.where(tf.equal(tscore, tf.reduce_max(tscore)), zero2, tscore)
    # gaussian
    tscore = tscore * tf.exp(-(ovr * ovr) / sigma)
    sigma = sigma
    # shape = tscore.get_shape().as_list()
    # tBDminus = tf.one_hot(maxscore_index, shape[0], dtype=tf.float32)
    # tBDminus= tf.concat([tBDminus,tBDminus,tBDminus,tBDminus],1)
    # tBD = tBD - tBD[maxscore_index] * tBDminus
    zero1 = tf.zeros(shape=tf.shape(tBD), dtype=tf.float32)

    tBD=tf.cast(tBD, dtype=tf.float32)

    index = tf.ones(shape=tf.shape(tscore), dtype=tf.int64)*maxscore_index

    tBD = tf.where(tf.equal(tBD, tBD[maxscore_index,:]), zero1, tBD)
    # tBDminus = tf.constant
    # tBD = tBD - tBDminus

    # keep = tf.where(tf.equal(keep, tscore[maxscore_index]), maxscore_index, keep)
    # keep = tf.cond(tf.equal(n, 0), body2, body3)
    # maxscore_index = tf.cast(maxscore_index, dtype=tf.float32)
    keep = tf.where(tf.equal(keep, n), index, keep)
    n = n + tf.cast(tf.constant(1), dtype=tf.int64)

    return keep, tBD, tscore, sigma, n

def py_cpu_softnms(dets, sc, sigma=0.3, Nt=0.3, thresh=0.001, method=2):
    """
    py_cpu_softnms
    :param dets:   boexs 坐标矩阵 format [y1, x1, y2, x2]
    :param sc:     每个 boxes 对应的分数
    :param Nt:     iou 交叠门限
    :param sigma:  使用 gaussian 函数的方差
    :param thresh: 最后的分数门限
    :param method: 使用的方法
    :return:       留下的 boxes 的 index
    """

    # indexes concatenate boxes with the last column
    # N = dets.shape[0].value
    # print(N)
    # indexes = tf.range(N)
    # dets = tf.concat(1,[dets, indexes.T])

    # the order of boxes coordinate is [y1,x1,y2,x2]
    scores = sc
    tBD = tf.identity(dets)
    tscore = tf.identity(scores)
    # A = tf.zeros(shape=tf.shape(tscore), dtype=tf.float32)
    # keep = tf.matrix_diag(A)
    keep = tf.transpose(tf.range(0,2000,1, dtype=tf.int64))
    n = tf.constant(0)
    n = tf.cast(n, dtype=tf.int64)
    keep, tBD, tscore, sigma, n= tf.while_loop(cond, body, [keep, tBD, tscore, sigma, n])
    # inds = tf.where(tscore > thresh)
    keep = keep[:100]
    # keep = keep.astype(int)
    print(keep)

    return keep






    # tBD = dets.copy()
    # tscore = scores.copy()
    # tarea = areas.copy()
    #
    # maxscore = tf.where(tf.not_equal(i, bv)
    # pos = i + 1
    # maxpos = tf.cond(, f1(pos), f2)
    # dets, tBD, scores, tscore, areas, tarea = tf.cond(tf.less(tscore, maxscore), f3(maxpos, tBD, tscore, tarea), )
    # xx1 = np.maximum(dets[i, 1], dets[pos:, 1])
    # yy1 = np.maximum(dets[i, 0], dets[pos:, 0])
    # xx2 = np.minimum(dets[i, 3], dets[pos:, 3])
    # yy2 = np.minimum(dets[i, 2], dets[pos:, 2])
    #
    # w = np.maximum(0.0, xx2 - xx1 + 1)
    # h = np.maximum(0.0, yy2 - yy1 + 1)
    # inter = w * h
    # ovr = inter / (areas[i] + areas[pos:] - inter)





    # def cond(i, N):
    #     return i < N
    # def f1(pos):
    #     maxscore = np.max(scores[pos:], axis=0)
    #     maxpos = np.argmax(scores[pos:], axis=0)
    #     return maxscore,maxpos
    #
    # def f2():
    #     maxscore = scores[-1]
    #     maxpos = 0
    #     return maxscore,maxpos
    #
    # def f3(maxpos, tBD, tscore, tarea):
    #     dets[i, :] = dets[maxpos + i + 1, :]
    #     dets[maxpos + i + 1, :] = tBD
    #     tBD = dets[i, :]
    #
    #     scores[i] = scores[maxpos + i + 1]
    #     scores[maxpos + i + 1] = tscore
    #     tscore = scores[i]
    #
    #     areas[i] = areas[maxpos + i + 1]
    #     areas[maxpos + i + 1] = tarea
    #     tarea = areas[i]
    #     return dets, tBD, scores, tscore, areas, tarea
    #
    # def f4(ovr):
    #     weight = np.ones(ovr.shape)
    #     weight[ovr > Nt] = weight[ovr > Nt] - ovr[ovr > Nt]
    #     return  weight
    # def f5(ovr):
    #     weight = np.exp(-(ovr * ovr) / sigma)
    #     return weight
    # def f6(ovr):
    #     weight = np.ones(ovr.shape)
    #     weight[ovr > Nt] = 0
    #     return weight

    # def body(dets, scores, areas, i, N):
    #     tBD = dets[i, :].copy()
    #     tscore = scores[i].copy()
    #     tarea = areas[i].copy()
        # pos = i + 1
        # maxscore, maxpos = tf.cond(tf.not_equal(i, N-1), f1(pos), f2)
        # dets, tBD, scores, tscore, areas, tarea = tf.cond(tf.less(tscore, maxscore), f3(maxpos, tBD, tscore, tarea), )
        # xx1 = np.maximum(dets[i, 1], dets[pos:, 1])
        # yy1 = np.maximum(dets[i, 0], dets[pos:, 0])
        # xx2 = np.minimum(dets[i, 3], dets[pos:, 3])
        # yy2 = np.minimum(dets[i, 2], dets[pos:, 2])
        #
        # w = np.maximum(0.0, xx2 - xx1 + 1)
        # h = np.maximum(0.0, yy2 - yy1 + 1)
        # inter = w * h
    #     # ovr = inter / (areas[i] + areas[pos:] - inter)
    #     tf.cond(tf.less(tscore, maxscore), f3(maxpos, tBD, tscore, tarea), )
    #     weight = tf.cond(np.equal(method, 1), f4(ovr), )
    #     weight = tf.cond(np.equal(method, 2), f5(ovr), )
    #     weight = tf.cond(np.equal(method, 3), f6(ovr), )
    #     scores[pos:] = weight * scores[pos:]
    #     i = i + 1
    #     return dets, scores, areas, i, N
    #
    # dets, scores, areas, i, N = tf.while_loop(cond, body, [dets, scores, areas, i, N])
    # inds = dets[:, 4][scores > thresh]
    # keep = inds.astype(int)
    # print(keep)
    #
    # return keep


    # for i in range(N):
    #     # intermediate parameters for later parameters exchange
    #     tBD = dets[i, :].copy()
    #     tscore = scores[i].copy()
    #     tarea = areas[i].copy()
    #     pos = i + 1
    #
    #     #
    #     if i != N-1:
    #         maxscore = np.max(scores[pos:], axis=0)
    #         maxpos = np.argmax(scores[pos:], axis=0)
    #     else:
    #         maxscore = scores[-1]
    #         maxpos = 0
    #     if tscore < maxscore:
    #         dets[i, :] = dets[maxpos + i + 1, :]
    #         dets[maxpos + i + 1, :] = tBD
    #         tBD = dets[i, :]
    #
    #         scores[i] = scores[maxpos + i + 1]
    #         scores[maxpos + i + 1] = tscore
    #         tscore = scores[i]
    #
    #         areas[i] = areas[maxpos + i + 1]
    #         areas[maxpos + i + 1] = tarea
    #         tarea = areas[i]

        # IoU calculate
        # xx1 = np.maximum(dets[i, 1], dets[pos:, 1])
        # yy1 = np.maximum(dets[i, 0], dets[pos:, 0])
        # xx2 = np.minimum(dets[i, 3], dets[pos:, 3])
        # yy2 = np.minimum(dets[i, 2], dets[pos:, 2])
        #
        # w = np.maximum(0.0, xx2 - xx1 + 1)
        # h = np.maximum(0.0, yy2 - yy1 + 1)
        # inter = w * h
        # ovr = inter / (areas[i] + areas[pos:] - inter)

        # Three methods: 1.linear 2.gaussian 3.original NMS
        # if method == 1:  # linear
        #     weight = np.ones(ovr.shape)
        #     weight[ovr > Nt] = weight[ovr > Nt] - ovr[ovr > Nt]
        # elif method == 2:  # gaussian
        #     weight = np.exp(-(ovr * ovr) / sigma)
        # else:  # original NMS
        #     weight = np.ones(ovr.shape)
        #     weight[ovr > Nt] = 0

        # scores[pos:] = weight * scores[pos:]

    # select the boxes and keep the corresponding indexes
    # inds = dets[:, 4][scores > thresh]
    # keep = inds.astype(int)
    # print(keep)
    #
    # return keep


# # boxes and scores
# boxes = np.array([[200, 200, 400, 400], [220, 220, 420, 420], [200, 240, 400, 440], [240, 200, 440, 400], [1, 1, 2, 2]], dtype=np.float32)
# boxscores = np.array([0.9, 0.8, 0.7, 0.6, 0.5], dtype=np.float32)
#
# # tf.image.non_max_suppression 中 boxes 是 [y1,x1,y2,x2] 排序的。
# with tf.Session() as sess:
#     # index = sess.run(tf.image.non_max_suppression(boxes=boxes, scores=boxscores, iou_threshold=0.5, max_output_size=5))
#     # print(index)
#     index = py_cpu_softnms(boxes, boxscores, method=3)
#     selected_boxes = sess.run(K.gather(boxes, index))
#     print(selected_boxes)