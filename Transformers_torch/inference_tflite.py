
import tensorflow as tf
tflite_interpreter = tf.lite.Interpreter(model_path="/home/parth/Intern/final_torch/mnist.tflite")
interpreter.allocate_tensors()


input_details = tflite_interpreter.get_input_details()

