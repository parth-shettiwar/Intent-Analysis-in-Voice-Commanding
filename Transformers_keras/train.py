import tensorlayer as tl
import argparse

import os
from itertools import islice
from typing import Iterable, List, Optional
from tensorflow.keras.layers import InputSpec

from keras import optimizers, losses
from keras.models import load_model
# noinspection PyPep8Naming
from keras import backend as K
from keras import callbacks
import numpy as np


from model import vanilla_transformer_gpt_model

# from model import Transformer
from data import dataset


def perplexity(y_true, y_pred):
  
    cross_entropy = K.sparse_categorical_crossentropy(y_true, y_pred)
    return K.mean(K.exp(K.mean(cross_entropy, axis=-1)))


total = 0
count=0
runs = 1
max_curr =  0
word_ids,labels,x_test,y_test,num_classes,seq_length,word_to_id = dataset("Dataset - Sheet3.csv")
# print(word_ids,num_tokens)
h,w = word_ids.shape
num_tokens = h*w
word_ids = word_ids[0:364]
j=0
optimizer = optimizers.Adam(
            lr=learning_rate, beta_1=0.9, beta_2=0.999, clipvalue=5.0)
model = vanilla_transformer_gpt_model(
            max_seq_length=6,
            vocabulary_size=2310,
            word_embedding_size=50,
            transformer_depth=1,
            num_heads=8)
model.compile(
            optimizer,
            loss=losses.sparse_categorical_crossentropy,
            metrics=[perplexity])

model.fit(
    word_ids, labels,
    batch_size=28, epochs=1)

# preds = model(word_ids[28*j:28*(j+1)])

    # final = torch.argmax(preds,axis = 1)
    
    # loss = criterion(preds,labels[28*j:28*(j+1)])
    #     # print("current step = ",i)
    # loss.backward()
    #     optimizer.step()
    #     running_loss += loss.item()
    #     if i % 100 == 99:    # print every 2000 mini-batches
    #         # print(running_loss / 100)
    #         # print(preds)
    #         running_loss = 0.0
    #     if(i%13==12):
    #         j=-1 
    #     j=j+1           
    


    # print(final)


#     test_preds = model(x_test)
#     test_final = torch.argmax(test_preds,axis = 1)
#     # print(y_test)
#     # print(test_final)
#     test_list = x_test.tolist()
#     inv_map = tl.nlp.build_reverse_dictionary(word_to_id)
#     inv_map.update({0:'okokok'})
#     for i in range(len(y_test)):
#         if(y_test[i]!=test_final[i]):
#             # print(tl.nlp.word_ids_to_words(test_list[i], inv_map))
#             # print(y_test[i])
#             # print(test_final[i])
#             count =  count+1
    
#     curr = acc = 100-((count/43)*100)
#     print("curr=",curr)
#     if(curr>max_curr):
#         PATH = './k20.pth'
#         torch.save(model, PATH)
#         max_curr = curr
#     total = total + count  
#     count=0  
#     print("run=",k)
# print(100-((total/(43*runs))*100))    