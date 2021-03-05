import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

# Functions for Transformer architecture
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

# Use PreLayerNorm for more stable training
class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, heads, p_drop=0.1):
        super(EncoderLayer, self).__init__()
        # multihead attention
        self.attn = nn.MultiheadAttention(d_model, heads, dropout=p_drop)
        # feedforward
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(p_drop)
        self.linear2 = nn.Linear(d_ff, d_model)

        # normalization module
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p_drop)
        self.dropout2 = nn.Dropout(p_drop)

    def forward(self, src):
        # Input shape for multihead attention: (SRCLEN, BATCH, EMB)
        # multihead attention w/ pre-LayerNorm
        src2 = self.norm1(src)
        src2, att_map = self.attn(src2, src2, src2) # projection to query, key, value are done in MultiheadAttention module
        src = src + self.dropout1(src2)

        # feed-forward
        src2 = self.norm2(src) # pre-normalization
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src, att_map

class Encoder(nn.Module):
    def __init__(self, enc_layer, n_layer, d_model):
        super(Encoder, self).__init__()
        self.layers = _get_clones(enc_layer, n_layer)
        self.n_layer = n_layer

    def forward(self, src):
        output = src
        for layer in self.layers:
            output, att = layer(output)
        return output, att

'''
class myTransformer(nn.Module):
    def __init__(self, n_layer=1, n_att_head=1, n_feat=128, p_drop=0.1):
        super(Pair2Pair, self).__init__()
        enc_layer_1 = EncoderLayer(d_model=n_feat, d_ff=n_feat*4,
                                   heads=n_att_head, p_drop=p_drop)
        self.encoder_1 = Encoder(enc_layer_1, n_layer, n_feat)
        enc_layer_2 = EncoderLayer(d_model=n_feat, d_ff=n_feat*4,
                                   heads=n_att_head, p_drop=p_drop)
        self.encoder_2 = Encoder(enc_layer_2, n_layer, n_feat)
        
    def forward(self, x):
        # Input: residue pair embeddings (L, L, C)
        # Ouput: residue pair embeddings (L, L, C)
        x, _ = self.encoder_1(x)
        x = x.permute(1,0,2)
        x, _ = self.encoder_2(x)
        x = x.permute(1,0,2)
        return x
'''
