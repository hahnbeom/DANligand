import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch.nn import Sequential, ReLU, GELU

from torch_geometric.nn import GATv2Conv, GATConv, GINConv, global_mean_pool, global_add_pool
from torch_geometric.nn import GINConv
import torch_scatter

# import sys
# sys.path.append('./custom_models/')

from custom_models.laf_model import LAFLayer

## LAF Aggregation Module ##
class GINLAFConv(GINConv):
    def __init__(self, nn, units=1, node_dim=32, **kwargs):
        super(GINLAFConv, self).__init__(nn, **kwargs)
        self.laf = LAFLayer(units=units, kernel_initializer='random_uniform')
        self.mlp = torch.nn.Linear(node_dim*units, node_dim)
        self.dim = node_dim
        self.units = units
    
    def aggregate(self, inputs, index):
        x = torch.sigmoid(inputs)
        x = self.laf(x, index)
        x = x.view((-1, self.dim * self.units))
        x = self.mlp(x)
        return x
    
## PNA Aggregation ##
class GINPNAConv(GINConv):
    def __init__(self, nn, node_dim=32, **kwargs):
        super(GINPNAConv, self).__init__(nn, **kwargs)
        self.mlp = torch.nn.Linear(node_dim*12, node_dim)
        self.delta = 2.5749
    
    def aggregate(self, inputs, index):
        sums = torch_scatter.scatter_add(inputs, index, dim=0)
        maxs = torch_scatter.scatter_max(inputs, index, dim=0)[0]
        means = torch_scatter.scatter_mean(inputs, index, dim=0)
        var = torch.relu(torch_scatter.scatter_mean(inputs ** 2, index, dim=0) - means ** 2)
        
        aggrs = [sums, maxs, means, var]
        c_idx = index.bincount().float().view(-1, 1)
        l_idx = torch.log(c_idx + 1.)
        
        amplification_scaler = [c_idx / self.delta * a for a in aggrs]
        attenuation_scaler = [self.delta / c_idx * a for a in aggrs]
        combinations = torch.cat(aggrs+ amplification_scaler+ attenuation_scaler, dim=1)
        x = self.mlp(combinations)
    
        return x
    

class LAFNet(torch.nn.Module):
    def __init__(self, dim_input, dim_output):
        super(LAFNet, self).__init__()

        num_features = dim_input
        dim = 32
        units = 3
        
        nn1 = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
        self.conv1 = GINLAFConv(nn1, units=units, node_dim=num_features)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv2 = GINLAFConv(nn2, units=units, node_dim=dim)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        nn3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv3 = GINLAFConv(nn3, units=units, node_dim=dim)
        self.bn3 = torch.nn.BatchNorm1d(dim)

        self.fc1 = Linear(dim, dim)
        self.fc2 = Linear(dim, dim_output)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = global_mean_pool(x, batch)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)


class PNANet(torch.nn.Module):
    def __init__(self, dim_input, dim_output):
        super(PNANet, self).__init__()

        num_features = dim_input
        dim = 64

        nn1 = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
        self.conv1 = GINPNAConv(nn1, node_dim=num_features)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv2 = GINPNAConv(nn2, node_dim=dim)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        nn3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv3 = GINPNAConv(nn3, node_dim=dim)
        self.bn3 = torch.nn.BatchNorm1d(dim)

        self.fc1 = Linear(dim, dim)
        self.fc2 = Linear(dim, dim_output)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = global_add_pool(x, batch)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)

## Simple GIN ##
class GINNet(torch.nn.Module):
    def __init__(self, dim_input, dim_output):
        super(GINNet, self).__init__()

        num_features = dim_input
        dim = 64

        nn1 = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        # nn3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        # self.conv3 = GINConv(nn3)
        # self.bn3 = torch.nn.BatchNorm1d(dim)

        self.fc1 = Linear(dim, dim)
        self.fc2 = Linear(dim, dim_output)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        # x = F.relu(self.conv3(x, edge_index))
        # x = self.bn3(x)
        x = global_mean_pool(x, batch)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return x


## Simple GAT ##
# using edge features
class GATNet(torch.nn.Module):
    def __init__(self, dim_input, dim_output, hid=4, num_in_head=8, num_out_head=1, edge_dim=3):
        super(GATNet, self).__init__()
        self.hid = hid
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.num_in_head = num_in_head
        self.num_out_head = num_out_head
        self.edge_dim = edge_dim

        self.conv1 = GATConv(self.dim_input, self.hid, num_in_head, edge_dim=self.edge_dim, dropout=0.1)
        self.conv2 = GATConv(self.num_in_head * self.hid, self.hid, num_in_head, edge_dim=self.edge_dim, dropout=0.1)
        self.conv3 = GATConv(self.num_in_head * self.hid, self.dim_output, concat=False, heads=self.num_out_head, edge_dim=self.edge_dim, dropout=0.1)

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.conv1(x, edge_index, edge_attr)
        x = F.elu(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = F.elu(x)
        x = self.conv3(x, edge_index, edge_attr)
        x = global_mean_pool(x, batch)
        return x


## Simple GAT (not using Edge features) ##
class GATNet_ori(torch.nn.Module):
    def __init__(self, dim_input, dim_output, hid=4, num_in_head=8, num_out_head=1):
        super(GATNet_ori, self).__init__()
        self.hid = hid
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.num_in_head = num_in_head
        self.num_out_head = num_out_head

        self.conv1 = GATConv(self.dim_input, self.hid, num_in_head, dropout=0.1)
        self.conv2 = GATConv(self.num_in_head * self.hid, self.dim_output, concat=False, heads=self.num_out_head, dropout=0.1)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)
        return x


## Simple GAT with MLP ##
class GATNet_MLP(torch.nn.Module):
    def __init__(self, dim_input, dim_output, hid=16,lin_hid=64, num_in_head=8, num_out_head=1, edge_dim=5):
        super(GATNet_MLP, self).__init__()
        self.hid = hid
        self.lin_hid = lin_hid
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.num_in_head = num_in_head
        self.num_out_head = num_out_head
        self.edge_dim = edge_dim

        self.conv1 = GATConv(self.dim_input, self.hid, num_in_head, edge_dim=self.edge_dim, dropout=0.1)
        self.conv2 = GATConv(self.num_in_head * self.hid, self.hid, num_in_head, edge_dim=self.edge_dim, dropout=0.1)
        self.conv3 = GATConv(self.num_in_head * self.hid, self.num_in_head * self.hid, concat=False, heads=self.num_out_head, edge_dim=self.edge_dim, dropout=0.1)

        self.fc_layers = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.num_in_head * self.hid, out_features=self.lin_hid),
            torch.nn.GELU(),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(in_features=self.lin_hid, out_features=self.dim_output)
            )

    def forward(self, x, edge_index, edge_attr, batch):
        # GAT
        x = self.conv1(x, edge_index, edge_attr)
        x = F.elu(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = F.elu(x)
        x = self.conv3(x, edge_index, edge_attr)
        x = F.elu(x)
        # FC layers
        x = self.fc_layers(x)
        x = global_mean_pool(x, batch)
        return x


## Simple GATv2 ##
class GATv2Net(torch.nn.Module):
    def __init__(self, dim_input, dim_output, hid=4, num_in_head=8, num_out_head=1):
        super(GATv2Net, self).__init__()
        self.hid = hid
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.num_in_head = num_in_head
        self.num_out_head = num_out_head

        self.conv1 = GATv2Conv(self.dim_input, self.hid, num_in_head, dropout=0.5)
        self.conv2 = GATv2Conv(self.num_in_head * self.hid, self.dim_output, concat=False, heads=self.num_out_head, dropout=0.5)

    def forward(self, x, edge_index, batch):
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)
        return x


