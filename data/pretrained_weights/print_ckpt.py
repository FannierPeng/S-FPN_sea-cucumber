#!/user/bin/env python
# _*_ coding:utf-8 _*_
#使用NewCheckpointReader来读取ckpt里的变量
from tensorflow.python import pywrap_tensorflow
import os
checkpoint_path = os.path.join("mobilenet_v1_1.0_224/mobilenet_v1_1.0_224.ckpt")
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path) #tf.train.NewCheckpointReader
var_to_shape_map = reader.get_variable_to_shape_map()
for key in var_to_shape_map:
  print("tensor_name: ", key)
  #print(reader.get_tensor(key))