import tensorlayer as tl
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

model_spec = coremltools.utils.load_spec("./myModel.mlmodel")
model_fp16_spec = coremltools.utils.convert_neural_network_spec_weights_to_fp16(model_spec)
coremltools.utils.save_spec(model_fp16_spec, 'k15red.mlmodel')
# quantized_model = quantization_utils.quantize_weights(model_spec, 16, "linear")
# coremltools.utils.save_spec(quantized_model, 'exampleModelFP8.mlmodel')
