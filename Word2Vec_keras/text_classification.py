

import numpy as np
import os
import numpy as np
import tensorflow as tf
assert tf.__version__.startswith('2')
from tensorflow.keras import layers

from text_dataloader import TextClassifierDataLoader
# from tensorflow_examples.lite.model_maker.core.task.model_spec import AverageWordVecModelSpec
# from tensorflow_examples.lite.model_maker.core.task.model_spec import BertClassifierModelSpec
import text_classifier
from word2vec import AverageWordVecModelSpec



# import zipfile
# with zipfile.ZipFile(os.path.abspath(os.getcwd()+"/train.zip"), 'r') as zip_ref:
#     zip_ref.extractall(os.path.abspath(os.getcwd())+"/train")
data_path =  os.path.abspath(os.getcwd())


# print(data_path)
model_spec = AverageWordVecModelSpec(seq_len=10,)


train_data1,idss = TextClassifierDataLoader.from_folder(os.path.join(data_path, 'train'),is_training=True, model_spec=model_spec, class_labels=['1', '2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26'])
print("###########################")
inv_map = {v: k for k, v in idss.items()}

train_data,test,ds = train_data1.split(0.9)
# test_data,validation_data,lss=test.split(0.5)
k=np.ones((1,10))
p=np.ones((1))
for i in ds:
  j=i[0].numpy()
  b=i[1].numpy()
  tt=inv_map[b]
  k=np.vstack((k,j))
  p=np.vstack((p,tt))
p=p[1:]
print(p)
print(p.shape)
print(p.dtype)
k=k[1:,:] 
k=k.astype(np.int32) 
p=p.astype(np.int32)
np.savetxt("labelstsne.txt",p)
k=tf.convert_to_tensor(k)
print(k)
ff=k
k=k.ref()
# print(k[1])
# for h in k:
#   print(h.ref())
#   print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
#   ff=tf.stack([ff,h.ref()])
# ff=ff[428:,:]  
model = text_classifier.create(train_data,ff, model_spec=model_spec,epochs=300,validation_data=test)

# loss, acc = model.evaluate(test_data)
# print(acc)
model.summary()


# from keras import backend as K

# # with a Sequential model

# from keras.models import Model

# layer_name = 'embedding'
# intermediate_layer_model = Model(inputs=k,
#                                  outputs=model.get_layer(layer_name).output)
# intermediate_output = intermediate_layer_model.predict(data)
model.export(export_dir='.')

