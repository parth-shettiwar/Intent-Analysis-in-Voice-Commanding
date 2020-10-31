import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import torch.optim as optim
import torch
from model import Transformer
# from model import Transformer
from data import dataset
total = 0
count=0
runs = 1
max_curr =  0
for k in range(runs):
    word_ids,labels,x_test,y_test,num_classes,seq_length,word_to_id = dataset("Dataset - Sheet3.csv")
    # print(word_ids,num_tokens)
    h,w = word_ids.shape
    num_tokens = h*w
    criterion = nn.CrossEntropyLoss()

    model = Transformer(k = 50, heads  = 8, depth=1, seq_length = seq_length, num_tokens = 2310, num_classes = num_classes)
    running_loss = 0.0
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    word_ids = word_ids[0:364]
    j=0
    
   

    for i in range(5000):
        optimizer.zero_grad()
        preds = model(word_ids[28*j:28*(j+1)])
        
        final = torch.argmax(preds,axis = 1)
    
        loss = criterion(preds,labels[28*j:28*(j+1)])
        # print("current step = ",i)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 100 == 99:    # print every 2000 mini-batches
            # print(running_loss / 100)
            # print(preds)
            running_loss = 0.0
        if(i%13==12):
            j=-1 
        j=j+1           
    


    # print(final)


    test_preds = model(x_test)
    test_final = torch.argmax(test_preds,axis = 1)
    # print(y_test)
    # print(test_final)
    test_list = x_test.tolist()
    inv_map = {v: k for k, v in word_to_id.items()}
    inv_map.update({0:'okokok'})
    for i in range(len(y_test)):
        if(y_test[i]!=test_final[i]):
            # print(tl.nlp.word_ids_to_words(test_list[i], inv_map))
            # print(y_test[i])
            # print(test_final[i])
            count =  count+1
    
    curr = acc = 100-((count/43)*100)
    print("curr=",curr)
    if(curr>max_curr):
        PATH = './k20.pth'
        torch.save(model, PATH)
        max_curr = curr
    total = total + count  
    count=0  
    print("run=",k)
print(100-((total/(43*runs))*100))    