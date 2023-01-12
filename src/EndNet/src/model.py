import torch
import torch.nn as nn

#from equivariant_attention.modules import get_basis_and_r, GSE3Res, GNormBias
#from equivariant_attention.modules import GConvSE3, GNormSE3
#from equivariant_attention.fibers import Fiber

from SE3.se3_transformer.model import SE3Transformer
from SE3.se3_transformer.model.fiber import Fiber
from src.src_Grid.model import SE3TransformerWrapper as Grid_SE3
from src.src_TR.model import SE3TransformerWrapper as TR_SE3


class SE3TransformerWrapper(nn.Module):
    """SE(3) equivariant GCN with attention"""
    def __init__(self, num_layers=2, num_channels=32, num_degrees=3, n_heads=4, div=4,
                 l0_in_features=32, l0_out_features=32,
                 l1_in_features=0, l1_out_features=8,
                 num_edge_features=32, ntypes=15,
                 drop_out=0.1,
                 n_trigonometry_module_stack=5,
                 bias=True):
        super().__init__()

        self.se3_Grid = Grid_SE3(
            num_layers=num_layers,
            l0_in_features=l0_in_features,
            num_edge_features=num_edge_features, #1-hot bond type x 2, distance
            l0_out_features=ntypes, #category only
            #l1_out_features=n_l1out,
	    ntypes=ntypes)

        self.se3_TR = TR_SE3( num_layers_lig=2,
                              num_layers_rec=2,
                              num_channels=num_channels,
                              num_degrees=num_degrees, n_heads_se3=4, div=4,
                              l0_in_features_lig=19,
                              l0_in_features_rec=ntypes,
                              l0_out_features=32,
                              l1_in_features=0,
                              l1_out_features=0, #???
                              K=4, # how many Y points
                              embedding_channels=32,
                              c=128,
                              n_trigonometry_module_stack = n_trigonometry_module_stack,
                              num_edge_features=5, #(bondtype-1hot x4, d) -- ligand only
                              dropout=0.1,
                              bias=True)

    def forward(self, G, node_features, Glig, labelidx, edge_features=None,):

        h_rec, cs = self.se3_Grid(G, node_features, edge_features)

        Yrec_s, z = self.se3_TR(G, h_rec, Glig, labelidx)

        return Yrec_s, z, cs #B x ? x ?
