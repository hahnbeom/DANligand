import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Attention_Layer(nn.Module):
    def __init__(self, 
                 l0_out_features=32,
                 nattn = 3,
                 bias=True):
        super().__init__()

        d = l0_out_features

        nattn = nattn


        self.gate_r = nn.Linear(2*d, 1)
        self.gate_l = nn.Linear(2*d, 1)
        
        self.W_r = nn.Linear(d,d)
        self.W_l = nn.Linear(d,d)

    def calc_attn(self, h_r_w, h_l_w):

        dots = torch.einsum("id,kd->ki",h_r_w,h_l_w) # K x N

        A = nn.functional.softmax(self.scale*dots,dim=1)


        h_r_prime = F.relu(torch.einsum('ki,id->id',A,h_r_w))
        h_l_prime = F.relu(torch.einsum('ki,kd->kd',A,h_l_w))



        zr = torch.sigmoid(self.gate_r(torch.cat([h_r_w, h_r_prime], -1))) # zr : nodenum x 1 
        zl = torch.sigmoid(self.gate_l(torch.cat([h_l_w, h_l_prime], -1)))
                    

        h_r_w = torch.einsum('ij,ik->ik',zr,h_r_w) + torch.einsum('ij,ik->ik',(1-zr),h_r_prime)
        h_l_w = torch.einsum('ij,ik->ik',zl,h_l_w) + torch.einsum('ij,ik->ik',(1-zl),h_l_prime) 

        return A, h_r_w, h_l_w



    def forward(self, h_r, h_l, nattn):
        
        h_r_w = self.W_r(h_r)
        h_l_w = self.W_l(h_l)

        for i_attn_layer in range(nattn):

            A, h_r_w, h_l_w = self.calc_attn( h_r_w, h_l_w )

        return A

class Gated_MultiHead_Attention(nn.Module):
    def __init__(self, l0_out_features=32, nheads = 8, nattn = 1):

        d = l0_out_features
        nattn = nattn

        super().__init__()

        self.attentions = [Attention_Layer( l0_out_features=d, nattn = nattn, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = Attention_Layer( concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)
