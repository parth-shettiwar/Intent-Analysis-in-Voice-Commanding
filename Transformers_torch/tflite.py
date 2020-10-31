import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import torch.optim as optim
import torch
import coremltools
from coremltools.models.neural_network import quantization_utils
import torch.onnx
from onnx_coreml import convert 
from model import Transformer
# from model import Transformer
from data import dataset

import tensorflow as tf
# make a converter object from the saved tensorflow file
import tensorflow as tf
import os

cmd = 'onnx-tf convert -i "model.onnx" -o  "model.pb"'



import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import torch.optim as optim
import torch
import torch.onnx
from model import Transformer
# from model import Transformer
from data import dataset
def _convert_softmax(builder, node, graph, err):
    '''
    convert to CoreML SoftMax ND Layer:
    https://github.com/apple/coremltools/blob/655b3be5cc0d42c3c4fa49f0f0e4a93a26b3e492/mlmodel/format/NeuralNetwork.proto#3547
    '''
    axis = node.attrs.get('axis', 1)
    builder.add_softmax_nd(
        name=node.name,
        input_name=node.inputs[0],
        output_name=node.outputs[0] + ('_softmax' if node.op_type == 'LogSoftmax' else ''),
        axis=axis
    )
    if node.op_type == 'LogSoftmax':
        builder.add_unary(
            name=node.name+'_log',
            input_name=node.outputs[0]+'_softmax',
            output_name=node.outputs[0],
            mode='log'
        )
model = torch.load("/home/parth/Intern/final_torch/k20.pth")
print(model)

dummy_input = torch.ones(28,6).long()
torch.onnx.export(model, dummy_input, "model.onnx", verbose=True,input_names = ['input'],output_names = ['output'])     
from onnx_coreml import convert 
coreml_model = convert("model.onnx",minimum_ios_deployment_target='13',custom_conversion_functions={'Softmax':_convert_softmax})
coreml_model.save("myModel.mlmodel") 

# converter = tf.lite.TFLiteConverter.from_saved_model("/home/parth/Intern/scratch",tags='None')
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
#                                        tf.lite.OpsSet.SELECT_TF_OPS]
# tflite_model = converter.convert()
# open("converted_model.tflite", "wb").write(tflite_model)

os.system(cmd)
converter = tf.lite.TFLiteConverter.from_frozen_graph('model.pb', #TensorFlow freezegraph .pb model file
                                                      input_arrays=['input'], # name of input arrays as defined in torch.onnx.export function before.
                                                      output_arrays=['output']  # name of output arrays defined in torch.onnx.export function before.
                                                      )
# tell converter which type of optimization techniques to use
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                       tf.lite.OpsSet.SELECT_TF_OPS]
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

# to view the best option for optimization read documentation of tflite about optimization
# go to this link https://www.tensorflow.org/lite/guide/get_started#4_optimize_your_model_optional
converter.experimental_new_converter = True

# convert the model 
tf_lite_model = converter.convert()
# save the converted model 
open('model.tflite', 'wb').write(tf_lite_model)
print("sssssssss")