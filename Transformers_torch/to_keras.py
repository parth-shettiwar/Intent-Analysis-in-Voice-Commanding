import tensorlayer as tl
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import torch.optim as optim
import torch
# import coremltools
# from coremltools.models.neural_network import quantization_utils
# import torch.onnx
# from onnx_coreml import convert 
# from model import Transformer
from pytorch2keras import pytorch_to_keras
# from model import Transformer
# from data import dataset
from torch.autograd import Variable

import tensorflow as tf
# make a converter object from the saved tensorflow file
import tensorflow as tf
model = torch.load("/home/parth/Intern/scratch/k15.pth")
# converter = tf.lite.TFLiteConverter.from_saved_model("/home/parth/Intern/scratch",tags='None')
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
#                                        tf.lite.OpsSet.SELECT_TF_OPS]
# tflite_model = converter.convert()
# open("converted_model.tflite", "wb").write(tflite_model)

# input_np = np.random.uniform(0, 1, (1, 10, 32, 32))
input_var = Variable(torch.ones(28,6))

k_model = pytorch_to_keras(model, input_var, [(28,6)], verbose=True)  