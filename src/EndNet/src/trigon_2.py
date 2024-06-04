from numpy import block
import torch
from torch import nn
from torch.nn import Linear
import sys
import torch.nn as nn
from torch.distributions import Categorical
#from torch_scatter import scatter_mean

class TriangleProteinToCompound_v2(torch.nn.Module):
    # separate left/right edges (block1/block2).
    def __init__(self, embedding_channels=256, c=128):
        super().__init__()
        self.layernorm = torch.nn.LayerNorm(embedding_channels)
        self.layernorm_c = torch.nn.LayerNorm(c)

        self.gate_linear1 = Linear(embedding_channels, c)
        self.gate_linear2 = Linear(embedding_channels, c)

        self.linear1 = Linear(embedding_channels, c)
        self.linear2 = Linear(embedding_channels, c)
        self.linear3 = Linear(c,c)
        self.linear4 = Linear(c,c)

        self.ending_gate_linear = Linear(embedding_channels, embedding_channels)
        self.linear_after_sum = Linear(c, embedding_channels)
        
    def forward(self, z, protein_pair, compound_pair, z_mask):
        # z of shape b, i, j, embedding_channels, where i is protein dim, j is compound dim.
        # z_mask : torch.Size([b, i, j, 1])

        z = self.layernorm(z)

        protein_pair = self.layernorm(protein_pair)
        compound_pair = self.layernorm(compound_pair)
 
        ab1 = self.gate_linear1(z).sigmoid() * self.linear1(z) * z_mask #torch.Size([b, i, j, c])
        ab2 = self.gate_linear2(z).sigmoid() * self.linear2(z) * z_mask

        protein_pair = self.gate_linear2(protein_pair).sigmoid() * self.linear2(protein_pair)
        compound_pair = self.gate_linear1(compound_pair).sigmoid() * self.linear1(compound_pair)

        g = self.ending_gate_linear(z).sigmoid()

        size_p = protein_pair.shape 
        size_c = compound_pair.shape 

        block1 = torch.einsum("bikc,bkjc->bijc", protein_pair, ab1)
        block2 = torch.einsum("bikc,bjkc->bijc", ab2, compound_pair)

        z = g * self.linear_after_sum(self.layernorm_c(block1+block2)) * z_mask
        return z

class Self_Attention(nn.Module):
    def __init__(self, hidden_size,num_attention_heads=8,drop_rate=0.5):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.dp = nn.Dropout(drop_rate)
        self.ln = nn.LayerNorm(hidden_size)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self,q,k,v,attention_mask=None,attention_weight=None):
        q = self.transpose_for_scores(q)
        k = self.transpose_for_scores(k)
        v = self.transpose_for_scores(v)
        attention_scores = torch.matmul(q, k.transpose(-1, -2))

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # attention_probs = self.dp(attention_probs)
        if attention_weight is not None:
            attention_weight_sorted_sorted = torch.argsort(torch.argsort(-attention_weight,axis=-1),axis=-1)
            # if self.training:
            #     top_mask = (attention_weight_sorted_sorted<np.random.randint(28,45))
            # else:
            top_mask = (attention_weight_sorted_sorted<32)
            attention_probs = attention_probs * top_mask
            # attention_probs = attention_probs * attention_weight
            attention_probs = attention_probs / (torch.sum(attention_probs,dim=-1,keepdim=True) + 1e-5)
        # print(attention_probs.shape,v.shape)
        # attention_probs = self.dp(attention_probs)
        outputs = torch.matmul(attention_probs, v)

        outputs = outputs.permute(0, 2, 1, 3).contiguous()
        new_output_shape = outputs.size()[:-2] + (self.all_head_size,)
        outputs = outputs.view(*new_output_shape)
        outputs = self.ln(outputs)
        return outputs

class TriangleSelfAttentionRowWise(torch.nn.Module):
    # use the protein-compound matrix only.
    def __init__(self, embedding_channels=128, c=32, num_attention_heads=4):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = c
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        # self.dp = nn.Dropout(drop_rate)
        # self.ln = nn.LayerNorm(hidden_size)

        self.layernorm = torch.nn.LayerNorm(embedding_channels)
        # self.layernorm_c = torch.nn.LayerNorm(c)

        self.linear_q = Linear(embedding_channels, self.all_head_size, bias=False)
        self.linear_k = Linear(embedding_channels, self.all_head_size, bias=False)
        self.linear_v = Linear(embedding_channels, self.all_head_size, bias=False)
        # self.b = Linear(embedding_channels, h, bias=False)
        self.g = Linear(embedding_channels, self.all_head_size)
        self.final_linear = Linear(self.all_head_size, embedding_channels)

    def reshape_last_dim(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x

    def forward(self, z, z_mask):
        # z of shape b, i, j, embedding_channels, where i is protein dim, j is compound dim.
        # z_mask of shape b, i, j
        z = self.layernorm(z)
        p_length = z.shape[1]
        batch_n = z.shape[0]
        # new_z = torch.zeros(z.shape, device=z.device)
        z_i = z
        z_mask_i = z_mask.view((batch_n, p_length, 1, 1, -1))
        attention_mask_i = (1e9 * (z_mask_i.float() - 1.))
        # q, k, v of shape b, j, h, c
        q = self.reshape_last_dim(self.linear_q(z_i)) #  * (self.attention_head_size**(-0.5))
        k = self.reshape_last_dim(self.linear_k(z_i))
        v = self.reshape_last_dim(self.linear_v(z_i))
        logits = torch.einsum('biqhc,bikhc->bihqk', q, k) + attention_mask_i
        weights = nn.Softmax(dim=-1)(logits)
        # weights of shape b, h, j, j
        # attention_probs = self.dp(attention_probs)
        weighted_avg = torch.einsum('bihqk,bikhc->biqhc', weights, v)
        g = self.reshape_last_dim(self.g(z_i)).sigmoid()
        output = g * weighted_avg
        new_output_shape = output.size()[:-2] + (self.all_head_size,)
        output = output.view(*new_output_shape)
        # output of shape b, j, embedding.
        # z[:, i] = output
        z = output
        # print(g.shape, block1.shape, block2.shape)
        z = self.final_linear(z) * z_mask.unsqueeze(-1)
        return z

class Transition(torch.nn.Module):
    # separate left/right edges (block1/block2).
    def __init__(self, embedding_channels=256, n=4, bias=True):
        super().__init__()
        self.layernorm = torch.nn.LayerNorm(embedding_channels)
        self.linear1 = Linear(embedding_channels, n*embedding_channels, bias=bias)
        self.linear2 = Linear(n*embedding_channels, embedding_channels, bias=bias)
    def forward(self, z):
        # z of shape b, i, j, embedding_channels, where i is protein dim, j is compound dim.
        z = self.layernorm(z)
        z = self.linear2((self.linear1(z)).relu())
        return z
