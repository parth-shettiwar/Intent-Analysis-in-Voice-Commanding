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