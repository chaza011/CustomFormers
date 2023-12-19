import torch
import torch.nn as nn
import torch.nn.functional as F
from .self_attention import MultiHeadAttBlockS

class Transformer(nn.Module):
    def __init__(self, d_model, d_ff, num_heads):
        super(Transformer, self).__init__()
        
        self.self_attention = MultiHeadAttBlockS(d_model, num_heads)
        
        self.hidden = nn.Linear(d_model, d_ff)
        self.output = nn.Linear(d_ff, d_model)
        self.GELU = nn.GELU()
        
        self.Dropout = nn.Dropout(p=0.1)
        
    def forward(self, x):
        
        #Attention
        self_attention = self.self_attention(x)
        
        #Hidden Layer
        hidden = self.GELU(self.hidden(self_attention))
        
        #Output Layer with res
        output = self.Dropout(self.output(hidden) + self_attention)
        
        return output
