# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import sys
sys.path.append('../../')
import xml.etree.cElementTree as ET
import numpy as np
import tensorflow as tf
import glob
import cv2
from help_utils.tools import *
from libs.label_name_dict.label_dict import *
import os

#重点改
# tf.app.flags.DEFINE_string('VOC_dir', cfgs.ROOT_PATH +'/data/sea cucumber/train/', 'Voc dir')
tf.app.flags.DEFINE_string('VOC_dir', os.path.abspath('../../..')+r'/DataSet/sea_cucumber/train/', 'Voc dir')
# tf.app.flags.DEFINE_string('VOC_dir', '/media/yangzl/TOSHIBA EXT/commonly_codes/FPN_Tensorflow-yangxue-sea cucumber/data/sea cucumber/train/', 'Voc dir')
tf.app.flags.DEFINE_string('xml_dir', 'Annotations', 'xml dir')
tf.app.flags.DEFINE_string('image_dir', 'JPEGImages', 'image dir')
tf.app.flags.DEFINE_string('save_name', 'train', 'save name')
tf.app.flags.DEFINE_string('save_dir', cfgs.ROOT_PATH + '/tfrecords/', 'save name')
tf.app.flags.DEFINE_string('img_format', '.jpg', 'format of image')
tf.app.flags.DEFINE_string('dataset', 'sea cucumber', 'dataset')
FLAGS = tf.app.flags.FLAGS


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def read_xml_gtbox_and_label(xml_path):

    """
    :param xml_path: the path of voc xml
    :return: a list contains gtboxes and labels, shape is [num_of_gtboxes, 5],
           and has [xmin, ymin, xmax, ymax, label] in a per row
    """

    tree = ET.parse(xml_path)
    root = tree.getroot()
    img_width = None
    img_height = None
    box_list = []
    for child_of_root in root:
        # if child_of_root.tag == 'filename':
        #     assert child_of_root.text == xml_path.split('/')[-1].split('.')[0] \
        #                                  + FLAGS.img_format, 'xml_name and img_name cannot match'

        if child_of_root.tag == 'size':
            for child_item in child_of_root:
                if child_item.tag == 'width':
                    img_width = int(child_item.text)
                if child_item.tag == 'height':
                    img_height = int(child_item.text)

        if child_of_root.tag == 'object':
            label = None
            for child_item in child_of_root:
                if child_item.tag == 'name':
                    label = NAME_LABEL_MAP[child_item.text]
                if child_item.tag == 'bndbox':
                    tmp_box = []
                    for node in child_item:
                        tmp_box.append(int(node.text))  # [x1, y1. x2, y2]
                    assert label is not None, 'label is none, error'
                    tmp_box.append(label)  # [x1, y1. x2, y2, label]
                    box_list.append(tmp_box)

    gtbox_label = np.array(box_list, dtype=np.int32)  # [x1, y1. x2, y2, label]

    xmin, ymin, xmax, ymax, label = gtbox_label[:, 0], gtbox_label[:, 1], gtbox_label[:, 2], gtbox_label[:, 3], \
                                    gtbox_label[:, 4]

    gtbox_label = np.transpose(np.stack([ymin, xmin, ymax, xmax, label], axis=0))  # [ymin, xmin, ymax, xmax, label]

    return img_height, img_width, gtbox_label


def convert_pascal_to_tfrecord():

    xml_path = FLAGS.VOC_dir + FLAGS.xml_dir
    image_path = FLAGS.VOC_dir + FLAGS.image_dir
    save_path = FLAGS.save_dir + FLAGS.dataset + '_' + FLAGS.save_name + '.tfrecord'
    mkdir(FLAGS.save_dir)

    # writer_options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)
    # writer = tf.python_io.TFRecordWriter(path=save_path, options=writer_options)
    writer = tf.python_io.TFRecordWriter(path=save_path)

    for count, xml in enumerate(glob.glob(xml_path + '/*.xml')):
        # to avoid path error in different development platform
        xml = xml.replace('\\', '/')

        # img_name = xml.split('/')[-1].split('.')[0] + FLAGS.img_format
        img_name = os.path.splitext(xml.split('/')[-1])[0] + FLAGS.img_format
        img_path = image_path + '/' + img_name

        if not os.path.exists(img_path):
            print('{} is not exist!'.format(img_path))
            continue
        print(img_name)
        img_height, img_width, gtbox_label = read_xml_gtbox_and_label(xml)

        # img = np.array(Image.open(img_path))
        img = cv2.imread(img_path)
        feature = tf.train.Features(feature={
            # maybe do not need encode() in linux
            'img_name': _bytes_feature(img_name.encode()),
            'img_height': _int64_feature(img_height),
            'img_width': _int64_feature(img_width),
            'img': _bytes_feature(img.tostring()),
            'gtboxes_and_label': _bytes_feature(gtbox_label.tostring()),
            'num_objects': _int64_feature(gtbox_label.shape[0])
        })

        example = tf.train.Example(features=feature)

        writer.write(example.SerializeToString())

        view_bar('Conversion progress', count + 1, len(glob.glob(xml_path + '/*.xml')))

    print('\nConversion is complete!')

if __name__ == '__main__':
    # xml_path = '../data/dataset/VOCdevkit/VOC2007/Annotations/000005.xml'
    # read_xml_gtbox_and_label(xml_path)

    convert_pascal_to_tfrecord()
