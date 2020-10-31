import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import torch.nn.functional as F

class SelfAttention(nn.Module):
  def __init__(self, k, heads=8):
    super().__init__()
    self.k, self.heads = k, heads
    self.tokeys    = nn.Linear(k, k * heads, bias=False)
    self.toqueries = nn.Linear(k, k * heads, bias=False)
    self.tovalues  = nn.Linear(k, k * heads, bias=False)

  
    self.unifyheads = nn.Linear(heads * k, k)
  def forward(self, x):
    b, t, k = x.size()
    h = self.heads

    queries = self.toqueries(x).view(b, t, h, k)
    keys    = self.tokeys(x)   .view(b, t, h, k)
    values  = self.tovalues(x) .view(b, t, h, k) 
    keys = keys.transpose(1, 2).contiguous().view(b * h, t, k)
    queries = queries.transpose(1, 2).contiguous().view(b * h, t, k)
    values = values.transpose(1, 2).contiguous().view(b * h, t, k) 
    queries = queries / (k ** (1/4))
    keys    = keys / (k ** (1/4))

    dot = torch.matmul(queries, keys.transpose(1, 2))

    dot = F.softmax(dot, dim=2) 
    out = torch.matmul(dot, values).view(b, h, t, k)
    out = out.transpose(1, 2).contiguous().view(b, t, h * k)
    return self.unifyheads(out)



class TransformerBlock(nn.Module):
  def __init__(self, k, heads):
    super().__init__()

    self.attention = SelfAttention(k, heads=heads)

    self.norm1 = nn.LayerNorm(k)
    self.norm2 = nn.LayerNorm(k)

    self.ff = nn.Sequential(
      nn.Linear(k, 4 * k),
      nn.ReLU(),
      nn.Linear(4 * k, k))

  def forward(self, x):
    attended = self.attention(x)
  
    x = self.norm1(attended + x)
    print(x.type())
    
    fedforward = self.ff(x)
    return self.norm2(fedforward + x)

class Transformer(nn.Module):
  def __init__(self, k, heads, depth, seq_length, num_tokens, num_classes):
      super().__init__()
      
      self.num_tokens = num_tokens
      self.token_emb = nn.Embedding(num_tokens, k)
     
      
      tblocks = []
      for i in range(depth):
          tblocks.append(TransformerBlock(k=k, heads=heads))
      self.tblocks = nn.Sequential(*tblocks)
      self.toprobs = nn.Linear(k, num_classes)
  def forward(self, x):
      
      x=x.long()
      tokens = self.token_emb(x)
      
      b, t, k = tokens.size()
     
      x=tokens
      print(x.type())
      x = self.tblocks(x)
      x = self.toprobs(x.mean(dim=1))
      return F.relu(x)