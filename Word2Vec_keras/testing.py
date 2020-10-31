# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Model specification."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import inspect
import os
import re
import tempfile

import tensorflow as tf
from exx.tensorflow_examples.lite.model_maker.core import compat
from text_dataloader import TextClassifierDataLoader
from exx.tensorflow_examples.lite.model_maker.core import file_util
# from exx.tensorflow_examples.lite.model_maker.core.task import hub_loader
from tensorflow import keras
from word2vec import AverageWordVecModelSpec
from tensorflow.keras import layers
# import tensorflow_hub as hub
# from tensorflow_hub import registry

from official.nlp import optimization

from official.nlp.bert import configs as bert_configs
# from official.nlp.bert import run_squad_helper
# from official.nlp.bert import squad_evaluate_v1_1
# from official.nlp.bert import squad_evaluate_v2_0
# from official.nlp.bert import tokenization

import numpy as np
from official.nlp.data import classifier_data_lib
from official.nlp.data import squad_lib

# from official.nlp.modeling import models
from official.utils.misc import distribution_utils
from sklearn.manifold import TSNE


with tf.io.gfile.GFile('model.tflite', 'rb') as f:
  model_content = f.read()
model_spec = AverageWordVecModelSpec(seq_len=10,)
# Read label names from label file.
with tf.io.gfile.GFile('labels.txt', 'r') as f:
  label_names = f.read().split('\n')




test_data = TextClassifierDataLoader.from_csv(filename="/home/parth/Skill based LU/Word2Vec_keras/data/test.csv", model_spec=model_spec, text_column = "input_text",label_column = "label",fieldnames = ["input_text","label"] ,is_training=False, shuffle=False)
# print(test_data)
# Initialze TensorFlow Lite inpterpreter.
interpreter = tf.lite.Interpreter(model_content=model_content)
interpreter.allocate_tensors()
input_index = interpreter.get_input_details()[0]['index']
output = interpreter.tensor(interpreter.get_output_details()[0]["index"])
# print(interpreter.get_input_details())
# Run predictions on each test data and calculate accuracy.
accurate_count = 0
for text, label in test_data.dataset:
    # Add batch dimension and convert to float32 to match with the model's input
    # data format.
    text = tf.expand_dims(text, 0)
    text = tf.cast(text, tf.float32)
    # print(text)
    # text=text.lower()
    # Run inference.
    interpreter.set_tensor(input_index, text)
    interpreter.invoke()

    # Post-processing: remove batch dimension and find the label with highest
    # probability.
    predict_label = np.argmax(output()[0])
    print(label.numpy())
    # Get label name with label index.
    predict_label_name = label_names[predict_label]
    accurate_count += (predict_label == label.numpy())
    if(predict_label != label.numpy()):
      print(text)
      print(predict_label)
      print(label)
    

accuracy = accurate_count * 1.0 / test_data.size
print('TensorFlow Lite model accuracy = %.4f' % accuracy)