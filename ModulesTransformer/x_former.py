import torch
import torch.nn as nn
import torch.nn.functional as F
from .cross_attention import MultiHeadAttBlockX

#X-Former block
class X_Former(nn.Module):
    def __init__(self, d_model, d_ff, num_heads):
        super(X_Former, self).__init__()
        
        self.LN_cross = nn.LayerNorm(d_model)
        
        self.self_attention = MultiHeadAttBlockS(d_model, num_heads)
        
        self.cross_attention = MultiHeadAttBlockX(d_model, num_heads)
        
        self.hidden = nn.Linear(d_model, d_ff)
        self.output = nn.Linear(d_ff, d_model)
        self.GELU = nn.GELU()
        
        self.Dropout = nn.Dropout(p=0.1)
        
    def forward(self, x, cond):
        
        #Norm cond
        cond = self.LN_cross(cond)
        
        #Attention
        self_attention = self.self_attention(x)
        cross_attention = self.cross_attention(self_attention, cond)
        
        #Hidden Layer
        hidden = self.GELU(self.hidden(cross_attention))
        
        #Output Layer with LN and skip
        output = self.Dropout(self.output(hidden) + cross_attention)
        
        return output
