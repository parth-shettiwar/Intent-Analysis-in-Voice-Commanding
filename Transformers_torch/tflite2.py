import numpy as np
import tensorflow as tf
print(tf.__version__)
# Load TFLite model and allocate tensors.
interpreter = tf.compat.v1.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(input_details)
print(output_details)
# Test model on random input data.
input_shape = input_details[0]['shape']
print(input_shape)
input_buf = np.random.randint(20, size=(28, 6))
# input_buf=input_buf.astype(float)
input_buf=np.array(input_buf,dtype=np.float32)
print(input_buf)
print(type(input_buf))
# input_len = np.array([1, 15], dtype=np.float32)
# input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_buf)


interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data.shape)
print(output_data)