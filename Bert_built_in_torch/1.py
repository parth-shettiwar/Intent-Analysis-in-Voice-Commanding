

import pandas as pd
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
train_df = pd.read_csv('Dataset - Sheet1.csv', header=None)

train_df['text'] = train_df.iloc[:,0] 
train_df['label'] = train_df.iloc[:,1] 
train_df = train_df.drop(train_df.columns[[0,1]], axis=1)
train_df['text'] = train_df['text'].apply(lambda x: x.replace('\\', ' '))
train_df['label'] = train_df['label'].apply(lambda x:x-1)
print(train_df)
#eval_df = pd.read_csv('data/test.csv', header=None)
#eval_df['text'] = eval_df.iloc[:, 1] + " " + eval_df.iloc[:, 2]
#eval_df = eval_df.drop(eval_df.columns[[1, 2]], axis=1)
#eval_df.columns = ['label', 'text']
#eval_df = eval_df[['text', 'label']]
#eval_df['text'] = eval_df['text'].apply(lambda x: x.replace('\\', ' '))
#eval_df['label'] = eval_df['label'].apply(lambda x:x-1)

from simpletransformers.classification import ClassificationModel

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')
# Create a ClassificationModel
# model = ClassificationModel('distilbert', 'distilbert-base-uncased', num_labels=26,use_cuda=False,args={'learning_rate':1e-5, 'num_train_epochs': 2, 'reprocess_input_data': True, 'overwrite_output_dir': True})
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-cased')

# self.args = {
#     "output_dir": "outputs/",
#     "cache_dir": "cache_dir/",
#     "fp16": True,
#     "fp16_opt_level": "O1",
#     "max_seq_length": 128,
#     "train_batch_size": 8,
#     "gradient_accumulation_steps": 1,
#     "eval_batch_size": 8,
#     "num_train_epochs": 1,
#     "weight_decay": 0,
#     "learning_rate": 4e-5,
#     "adam_epsilon": 1e-8,
#     "warmup_ratio": 0.06,
#     "warmup_steps": 0,
#     "max_grad_norm": 1.0,

#     "logging_steps": 50,
#     "save_steps": 2000,

#     "overwrite_output_dir": False,
#     "reprocess_input_data": False,
#     "evaluate_during_training": False,

#     "process_count": 1,
#     "n_gpu": 1,
# }
import numpy as np
a = train_df['text'].to_list()
npa = np.asarray(a, dtype=np.float32)

# Create a TransformerModel with modified attributes
# model = TransformerModel('roberta', 'roberta-base', num_labels=4, args={'learning_rate':1e-5, 'num_train_epochs': 2, 'reprocess_input_data': True, 'overwrite_output_dir': True})
input_ids = torch.tensor(tokenizer.encode(npa, add_special_tokens=True)).unsqueeze(0)
# Train the model
model(a,train_df['label'])

