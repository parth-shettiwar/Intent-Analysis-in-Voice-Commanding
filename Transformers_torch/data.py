import re
import pandas as pd
import torch
from gensim.parsing.preprocessing import remove_stopwords
from gensim.parsing.preprocessing import STOPWORDS
from sklearn.model_selection import train_test_split
import os
from collections import Counter
import numpy as np
all_stopwords_gensim = STOPWORDS
sw_list = {"next","Next"}
all_stopwords_gensim = STOPWORDS.union(set(['insert','Insert','inserting','format','Format,formating','formatting'])).difference(sw_list)
# all_stopwords_gensim = STOPWORDS.difference(sw_list)

def dataset(name):
    data_path = os.getcwd() + '/data'
    train_path = os.path.join(data_path, name)
    train_df = pd.read_csv(train_path, header=None)
    train_df['text'] = train_df.iloc[:,0] 
    train_df['label'] = train_df.iloc[:,1] 
    train_df = train_df.drop(train_df.columns[[0,1]], axis=1)
    train_df['text'] = train_df['text'].apply(lambda x: x.replace('\\', ' '))
    train_df['label'] = train_df['label'].apply(lambda x:x-1)

    input_text = train_df['text'].to_list()
    for i in range(len(input_text)):
        input_text[i] = [word for word in input_text[i].split() if not word in all_stopwords_gensim]
        input_text[i] = " ".join(input_text[i])

    labels = train_df['label'].to_list()
    labels2 = np.asarray(labels)
    bag_of_words = []
  
    input_text_lo = input_text
    max_len = 0
    #make_dictionary
    for i in range(len(input_text)):
        input_text_lo[i] = input_text[i].lower()
        bag_of_words = bag_of_words + re.sub("[^\w]", " ",  input_text_lo[i]).split()
        if(len(re.sub("[^\w]", " ",  input_text_lo[i]).split())>max_len):
            max_len = len(re.sub("[^\w]", " ",  input_text_lo[i]).split())
    
    word_to_id = dict(zip(bag_of_words,range(len(bag_of_words))))
    for i, (k, v) in enumerate(word_to_id.items()):
        word_to_id[k]=i+1

    print(word_to_id)
    # for key,val in word_to_id.items():
    #     word_to_id[key] = word_to_id[key] + 1
    word_ids = np.zeros((len(input_text_lo),max_len))
    for i in range(len(input_text_lo)):
        temp = re.sub("[^\w]", " ",  input_text_lo[i]).split()
        for j in range(len(re.sub("[^\w]", " ",  input_text_lo[i]).split())):
           word_ids[i][j] =  word_to_id[temp[j]]  
    word_ids_train,word_ids_test,labels_train,labels_test= train_test_split(word_ids,labels2,test_size = .1,shuffle = True)        
    
    
    word_ids_train = torch.from_numpy(word_ids_train.astype(float))
    word_ids_test = torch.from_numpy(word_ids_test.astype(float))
    labels_train = torch.from_numpy(labels_train.astype(int))
    labels_test = torch.from_numpy(labels_test.astype(int))    

    
    return word_ids_train,labels_train,word_ids_test,labels_test,len(Counter(labels).keys()),max_len,word_to_id        

