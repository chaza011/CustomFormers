import torch
import torch.nn as nn
import torch.nn.functional as F
import math

#Attention Mechanism
def attention(q, k, v, d_k):
    
    attention_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    
    attention_weights = F.softmax(attention_scores, dim=-1)
    
    output = torch.matmul(attention_weights, v)
    
    return output

#Multi-head Cross-Attention Block
class MultiHeadAttBlockX(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttBlockX, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.Dropout = nn.Dropout(p=0.1)
        
        self.Q_linear = nn.Linear(d_model, d_model)
        self.K_linear = nn.Linear(d_model, d_model)
        self.V_linear = nn.Linear(d_model, d_model)
        
        self.output_linear = nn.Linear(d_model, d_model)
        
        self.LN_pre = nn.LayerNorm(d_model)
        self.LN_post = nn.LayerNorm(d_model)

    def forward(self, x, cond):
        batch_size, seq_length_x, _ = x.size()
        _, seq_length_cond, _ = cond.size()
        
        x = self.LN_pre(x)

        #Generate Q, K, V for all heads
        Q = self.Q_linear(x).view(batch_size, seq_length_x, self.num_heads, self.head_dim)
        K = self.K_linear(cond).view(batch_size, seq_length_cond, self.num_heads, self.head_dim)
        V = self.V_linear(cond).view(batch_size, seq_length_cond, self.num_heads, self.head_dim)

        #Transpose for head dimension
        Q, K, V = [matrix.transpose(1, 2) for matrix in [Q, K, V]]

        #Apply each attention head to its segment
        head_outputs = attention(Q, K, V, self.head_dim)

        #Reshape back to original dimension
        concatenated = head_outputs.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
        #Send through output layer and Apply LayerNorm and Res Connection
        output_attention = self.LN_post(self.Dropout(self.output_linear(concatenated) + x))
        
        return output_attention

        
