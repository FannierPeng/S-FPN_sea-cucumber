# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(1234)
import sys
sys.path.append('../')
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
import time
from data.io.read_tfrecord import next_batch
from libs.networks.network_factory import get_flags_byname
from libs.networks.network_factory import get_network_byname
from libs.configs import cfgs
from libs.rpn import build_rpn
from libs.fast_rcnn import build_fast_rcnn
from help_utils.tools import *
from libs.box_utils.show_box_in_tensor import draw_boxes_with_categories, draw_box_with_color
from tools import restore_model
from tools import read_npy, valval
import pandas as pd
FLAGS = get_flags_byname(cfgs.NET_NAME)
os.environ["CUDA_VISIBLE_DEVICES"] = cfgs.GPU_GROUP

def _concact_features(conv_output):
    """
    对特征图进行reshape拼接
    :param conv_output:输入多通道的特征图
    :return:
    """
    num_or_size_splits = conv_output.get_shape().as_list()[-1]
    each_convs = tf.split(conv_output, num_or_size_splits=num_or_size_splits, axis=3)
    concact_size = int(math.sqrt(num_or_size_splits) / 1)
    all_concact = None
    for i in range(concact_size):
        row_concact = each_convs[i * concact_size]
        for j in range(concact_size - 1):
            row_concact = tf.concat([row_concact, each_convs[i * concact_size + j + 1]], 1)
        if i == 0:
            all_concact = row_concact
        else:
            all_concact = tf.concat([all_concact, row_concact], 2)

    return all_concact

def train():
    with tf.Graph().as_default():
        tf.set_random_seed(1234)
        with tf.name_scope('get_batch'):
            img_name_batch, img_batch, gtboxes_and_label_batch, num_objects_batch = \
                next_batch(dataset_name=cfgs.DATASET_NAME,
                           batch_size=cfgs.BATCH_SIZE,
                           shortside_len=cfgs.SHORT_SIDE_LEN,
                           is_training=True,
                           is_val=False)

        with tf.name_scope('draw_gtboxes'):
            gtboxes_in_img = draw_box_with_color(img_batch, tf.reshape(gtboxes_and_label_batch, [-1, 5])[:, :-1],
                                                 text=tf.shape(gtboxes_and_label_batch)[1])


        # *********************************************************************
        # *                                         share net                                           *
        # ***********************************************************************************************
        _, share_net = get_network_byname(net_name=cfgs.NET_NAME,
                                          inputs=img_batch,
                                          num_classes=None,
                                          is_training=True,
                                          output_stride=None,
                                          global_pool=False,
                                          spatial_squeeze=False)

        # ***********************************************************************************************
        # *                                            rpn                                              *
        # ***********************************************************************************************
        rpn = build_rpn.RPN(net_name=cfgs.NET_NAME,
                            inputs=img_batch,
                            gtboxes_and_label=tf.squeeze(gtboxes_and_label_batch, 0),
                            is_training=True,
                            share_head=cfgs.SHARE_HEAD,
                            share_net=share_net,
                            stride=cfgs.STRIDE,
                            anchor_ratios=cfgs.ANCHOR_RATIOS,
                            anchor_scales=cfgs.ANCHOR_SCALES,
                            scale_factors=cfgs.SCALE_FACTORS,
                            base_anchor_size_list=cfgs.BASE_ANCHOR_SIZE_LIST,  # P2, P3, P4, P5, P6
                            level=cfgs.LEVEL,
                            top_k_nms=cfgs.RPN_TOP_K_NMS,
                            rpn_nms_iou_threshold=cfgs.RPN_NMS_IOU_THRESHOLD,
                            max_proposals_num=cfgs.MAX_PROPOSAL_NUM,
                            rpn_iou_positive_threshold=cfgs.RPN_IOU_POSITIVE_THRESHOLD,
                            rpn_iou_negative_threshold=cfgs.RPN_IOU_NEGATIVE_THRESHOLD,  # iou>=0.7 is positive box, iou< 0.3 is negative
                            rpn_mini_batch_size=cfgs.RPN_MINIBATCH_SIZE,
                            rpn_positives_ratio=cfgs.RPN_POSITIVE_RATE,
                            remove_outside_anchors=False,  # whether remove anchors outside
                            rpn_weight_decay=cfgs.WEIGHT_DECAY[cfgs.NET_NAME])

        rpn_proposals_boxes, rpn_proposals_scores = rpn.rpn_proposals()  # rpn_score shape: [300, ]

        rpn_location_loss, rpn_classification_loss = rpn.rpn_losses()
        rpn_total_loss = rpn_classification_loss + rpn_location_loss

        with tf.name_scope('draw_proposals'):
            # score > 0.5 is object
            rpn_object_boxes_indices = tf.reshape(tf.where(tf.greater(rpn_proposals_scores, 0.5)), [-1])
            rpn_object_boxes = tf.gather(rpn_proposals_boxes, rpn_object_boxes_indices)

            rpn_proposals_objcet_boxes_in_img = draw_box_with_color(img_batch, rpn_object_boxes,
                                                                    text=tf.shape(rpn_object_boxes)[0])
            rpn_proposals_boxes_in_img = draw_box_with_color(img_batch, rpn_proposals_boxes,
                                                             text=tf.shape(rpn_proposals_boxes)[0])
        # ***********************************************************************************************
        # *                                         Fast RCNN                                           *
        # ***********************************************************************************************

        fast_rcnn = build_fast_rcnn.FastRCNN(img_batch=img_batch,
                                             feature_pyramid=rpn.feature_pyramid,
                                             rpn_proposals_boxes=rpn_proposals_boxes,
                                             rpn_proposals_scores=rpn_proposals_scores,
                                             img_shape=tf.shape(img_batch),
                                             roi_size=cfgs.ROI_SIZE,
                                             roi_pool_kernel_size=cfgs.ROI_POOL_KERNEL_SIZE,
                                             scale_factors=cfgs.SCALE_FACTORS,
                                             gtboxes_and_label=tf.squeeze(gtboxes_and_label_batch, 0),
                                             fast_rcnn_nms_iou_threshold=cfgs.FAST_RCNN_NMS_IOU_THRESHOLD,
                                             fast_rcnn_maximum_boxes_per_img=100,
                                             fast_rcnn_nms_max_boxes_per_class=cfgs.FAST_RCNN_NMS_MAX_BOXES_PER_CLASS,
                                             show_detections_score_threshold=cfgs.FINAL_SCORE_THRESHOLD,  # show detections which score >= 0.6
                                             num_classes=cfgs.CLASS_NUM,
                                             fast_rcnn_minibatch_size=cfgs.FAST_RCNN_MINIBATCH_SIZE,
                                             fast_rcnn_positives_ratio=cfgs.FAST_RCNN_POSITIVE_RATE,
                                             fast_rcnn_positives_iou_threshold=cfgs.FAST_RCNN_IOU_POSITIVE_THRESHOLD,  # iou>0.5 is positive, iou<0.5 is negative
                                             use_dropout=False,
                                             weight_decay=cfgs.WEIGHT_DECAY[cfgs.NET_NAME],
                                             is_training=True,
                                             level=cfgs.LEVEL)

        fast_rcnn_decode_boxes, fast_rcnn_score, num_of_objects, detection_category = \
            fast_rcnn.fast_rcnn_predict()
        fast_rcnn_location_loss, fast_rcnn_classification_loss = fast_rcnn.fast_rcnn_loss()
        fast_rcnn_total_loss = fast_rcnn_location_loss + fast_rcnn_classification_loss

        with tf.name_scope('draw_boxes_with_categories'):
            fast_rcnn_predict_boxes_in_imgs = draw_boxes_with_categories(img_batch=img_batch,
                                                                         boxes=fast_rcnn_decode_boxes,
                                                                         labels=detection_category,
                                                                         scores=fast_rcnn_score)

        # train
        total_loss = slim.losses.get_total_loss()

        global_step = slim.get_or_create_global_step()#返回并创建全局步长张量
        #
        # lr = tf.train.piecewise_constant(global_step,
        #                                  boundaries=[np.int64(10000), np.int64(20000)],
        #                                  values=[cfgs.LR, cfgs.LR / 10, cfgs.LR / 100])
        lr = tf.train.exponential_decay(cfgs.LR, global_step, decay_steps=5000, decay_rate=1/2., staircase=True)
        # lr = tf.train.piecewise_constant(global_step,
        #                                  boundaries=[np.int64(30000), np.int64(40000)],
        #                                  values=[lr, cfgs.LR/100, cfgs.LR/1000])
        tf.summary.scalar('learning_rate', lr)
        # optimizer = tf.train.MomentumOptimizer(lr, momentum=cfgs.MOMENTUM)
        optimizer = tf.train.AdamOptimizer(lr,beta1=cfgs.MOMENTUM,beta2=0.999,epsilon=1e-8,use_locking=False,name='Adam')
        # optimizer = tf.train.RMSPropOptimizer(lr, decay=0.9, epsilon=1e-6, name='RMSProp')
        #创建一个计算梯度并返回损失的Operation
        train_op = slim.learning.create_train_op(total_loss, optimizer, global_step)  # rpn_total_loss,
        # train_op = optimizer.minimize(second_classification_loss, global_step)

        # ***********************************************************************************************
        # *                                          Summary                                            *
        # ***********************************************************************************************
        # ground truth and predict
        tf.summary.image('img/gtboxes', gtboxes_in_img)
        tf.summary.image('img/faster_rcnn_predict', fast_rcnn_predict_boxes_in_imgs)
        # rpn loss and image
        tf.summary.scalar('rpn/rpn_location_loss', rpn_location_loss)
        tf.summary.scalar('rpn/rpn_classification_loss', rpn_classification_loss)
        tf.summary.scalar('rpn/rpn_total_loss', rpn_total_loss)

        tf.summary.scalar('fast_rcnn/fast_rcnn_location_loss', fast_rcnn_location_loss)
        tf.summary.scalar('fast_rcnn/fast_rcnn_classification_loss', fast_rcnn_classification_loss)
        tf.summary.scalar('fast_rcnn/fast_rcnn_total_loss', fast_rcnn_total_loss)

        tf.summary.scalar('loss/total_loss', total_loss)
        # #
        # tf.summary.image('C2', _concact_features(share_net['resnet_v1_50/block1/unit_2/bottleneck_v1'][:, :, :, 0:16]), 1)
        # tf.summary.image('C3', _concact_features(share_net['resnet_v1_50/block2/unit_3/bottleneck_v1'][:, :, :, 0:16]), 1)
        # tf.summary.image('C4', _concact_features(share_net['resnet_v1_50/block3/unit_5/bottleneck_v1'][:, :, :, 0:16]), 1)
        # tf.summary.image('C5', _concact_features(share_net['resnet_v1_50/block4'][:, :, :, 0:16]), 1)
        # tf.summary.image('P2', _concact_features(rpn.feature_pyramid['P2'][:, :, :, 0:16]),1)
        # tf.summary.image('P3', _concact_features(rpn.feature_pyramid['P3'][:, :, :, 0:16]),1)
        # tf.summary.image('P4', _concact_features(rpn.feature_pyramid['P4'][:, :, :, 0:16]),1)
        # tf.summary.image('P5', _concact_features(rpn.feature_pyramid['P5'][:, :, :, 0:16]), 1)
        # tf.summary.image('rpn/rpn_all_boxes', rpn_proposals_boxes_in_img)
        # tf.summary.image('rpn/rpn_object_boxes', rpn_proposals_objcet_boxes_in_img)
        # learning_rate
        # tf.summary.scalar('learning_rate', lr)

        summary_op = tf.summary.merge_all()
        init_op = tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer()
        )

        restorer, restore_ckpt = restore_model.get_restorer()

        saver = tf.train.Saver(max_to_keep=16)

        config = tf.ConfigProto()
        # config.gpu_options.per_process_gpu_memory_fraction = 0.5
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:

            if cfgs.NET_NAME == 'pvanet':
                sess.run(init_op)
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess, coord)
                start = 0
                if not restorer is None:
                    restorer.restore(sess, restore_ckpt)
                    print('restore model')
                    start = int("".join(list(restore_ckpt.split('/')[-1])[4:8]))+1
                else:
                    # read_npy.load_initial_weights(sess)
                    read_npy.load_ckpt_weights(sess)
            else:

                sess.run(init_op)
                # print(sess.run('resnet_v1_50/block4/unit_3/bottleneck_v1/conv3/BatchNorm/moving_variance'))
                # print(sess.run('vgg_16/block4/unit_3/bottleneck_v1/conv3/BatchNorm/moving_variance'))
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess, coord)
                start = 0
                if not restorer is None:
                    restorer.restore(sess, restore_ckpt)
                    print('restore model')
                    # start = int("".join(list(restore_ckpt.split('/')[-1])[4:8]))+1


            summary_path = os.path.join(FLAGS.summary_path, cfgs.VERSION)
            mkdir(summary_path)
            summary_writer = tf.summary.FileWriter(summary_path, graph=sess.graph)
            df = pd.DataFrame([],columns=['Recall', 'Precision', 'mAP', 'F1_score'],index=[])

            for step in range(0,cfgs.MAX_ITERATION):
                # print(img_name_batch.eval())
                training_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
                start = time.time()

                _global_step, _img_name_batch, _rpn_location_loss, _rpn_classification_loss, \
                _rpn_total_loss, _fast_rcnn_location_loss, _fast_rcnn_classification_loss, \
                _fast_rcnn_total_loss, _total_loss, _ = \
                    sess.run([global_step, img_name_batch, rpn_location_loss, rpn_classification_loss,
                              rpn_total_loss, fast_rcnn_location_loss, fast_rcnn_classification_loss,
                              fast_rcnn_total_loss, total_loss, train_op])

                end = time.time()
                # if step == 100:
                #     save_dir = os.path.join(FLAGS.trained_checkpoint, cfgs.VERSION)
                #     mkdir(save_dir)
                #
                #     save_ckpt = os.path.join(save_dir, 'voc_' + str(_global_step) + 'model.ckpt')
                #     saver.save(sess, save_ckpt)
                #     print(' weights had been saved')
                # if step == 500:
                #     save_dir = os.path.join(FLAGS.trained_checkpoint, cfgs.VERSION)
                #     mkdir(save_dir)
                #
                #     save_ckpt = os.path.join(save_dir, 'voc_' + str(_global_step) + 'model.ckpt')
                #     saver.save(sess, save_ckpt)
                #     print(' weights had been saved')
                if step % 50 == 0:
                    print(""" {}: step{}    image_name:{} |\t
                                rpn_loc_loss:{} |\t rpn_cla_loss:{} |\t rpn_total_loss:{} |
                                fast_rcnn_loc_loss:{} |\t fast_rcnn_cla_loss:{} |\t fast_rcnn_total_loss:{} |
                                total_loss:{} |\t pre_cost_time:{}s""" \
                          .format(training_time, _global_step, str(_img_name_batch[0]), _rpn_location_loss,
                                  _rpn_classification_loss, _rpn_total_loss, _fast_rcnn_location_loss,
                                  _fast_rcnn_classification_loss, _fast_rcnn_total_loss, _total_loss,
                                  (end - start)))
                    # print(""" {}: step{}    image_name:{} |\t
                    #             rpn_loc_loss:{} |\t
                    #             fast_rcnn_loc_loss:{} |\t fast_rcnn_cla_loss:{} |\t fast_rcnn_total_loss:{} |
                    #             total_loss:{} |\t pre_cost_time:{}s""" \
                    #       .format(training_time, _global_step, str(_img_name_batch[0]), _rpn_location_loss,
                    #                _fast_rcnn_location_loss,
                    #               _fast_rcnn_classification_loss, _fast_rcnn_total_loss, _total_loss,
                    #               (end - start)))

                if step % 250 == 0:
                    summary_str = sess.run(summary_op)
                    summary_writer.add_summary(summary_str, _global_step)
                    summary_writer.flush()

                if (step > 0 and step% 2000 == 0) or (step > 0 and (step == 1000))or(step == cfgs.MAX_ITERATION - 1):
                    save_dir = os.path.join(FLAGS.trained_checkpoint, cfgs.VERSION)
                    mkdir(save_dir)

                    save_ckpt = os.path.join(save_dir, 'voc_'+str(_global_step)+'model.ckpt')
                    saver.save(sess, save_ckpt)
                    print(' weights had been saved')
                #保存验证集信息
                if (step > 0 and step% 2000 == 0) or (step == cfgs.MAX_ITERATION - 1):
                    save_excel = os.path.abspath('../') + r'/Loss/' + cfgs.NET_NAME + r'_' + cfgs.VERSION
                    mkdir(save_excel)

                    new_index = np.append(df.index, [str(step)])
                    df2 = pd.DataFrame([valval.val(is_val=True)], columns=['Recall', 'Precision', 'mAP', 'F1_score'])
                    df = df.append(df2)
                    df.index = new_index

                    df.to_excel(save_excel + r'/validation.xls')
                    print('validation result had been saved')


            coord.request_stop()
            coord.join(threads)


if __name__ == '__main__':

    train()


















