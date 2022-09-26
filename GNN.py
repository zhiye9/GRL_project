import numpy as np
import networkx as nx
from networkx.algorithms import centrality as cen
from networkx.algorithms import efficiency_measures as eff
from torch_geometric.utils import remove_self_loops
from torch_sparse import coalesce
from torch_geometric.data import Dataset, Data
from sklearn.model_selection import StratifiedKFold
from torch_geometric.nn import GCNConv, PNAConv, TransformerConv, GATConv
from sklearn.metrics import roc_auc_score
import torch
import torch.nn.functional as F
import pandas as pd
import os
from tqdm import tqdm
from torch_geometric.nn import global_add_pool, global_mean_pool

class GATN6(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim,
                 dropout, return_embeds=False):
        
        super(GCN, self).__init__()

        self.gcn1 = GATConv(input_dim, hidden_dim)
        self.gcn2 = GATConv(hidden_dim, hidden_dim)
        self.gcn3 = GATConv(hidden_dim, hidden_dim)
        self.gcn4 = GATConv(hidden_dim, hidden_dim)
        self.gcn5 = GATConv(hidden_dim, hidden_dim)
        self.gcn6 = GATConv(hidden_dim, output_dim)         
        self.softmax = torch.nn.LogSoftmax()

        self.dropout = dropout
        self.return_embeds = return_embeds 
        

    def reset_parameters(self):
        self.gcn1.reset_parameters()
        self.gcn2.reset_parameters()
        self.gcn3.reset_parameters()
        self.gcn4.reset_parameters()
        self.gcn5.reset_parameters()
        self.gcn6.reset_parameters()

    def forward(self, x, edge_index, edge_weight):

        x1 = F.relu(self.gcn1(x, edge_index, edge_weight))
        if self.training:
            x1 = F.dropout(x1, p=self.dropout)

        x1 = F.relu(self.gcn2(x1, edge_index, edge_weight))
        if self.training:
            x1 = F.dropout(x1, p=self.dropout)

        x1 = F.relu(self.gcn3(x1, edge_index, edge_weight))
        if self.training:
            x1 = F.dropout(x1, p=self.dropout)

        x1 = F.relu(self.gcn4(x1, edge_index, edge_weight))
        if self.training:
            x1 = F.dropout(x1, p=self.dropout)

        x1 = F.relu(self.gcn5(x1, edge_index, edge_weight))
        if self.training:
            x1 = F.dropout(x1, p=self.dropout)

        x1 = F.relu(self.gcn6(x1, edge_index, edge_weight))
        if self.training:
            x1 = F.dropout(x1, p=self.dropout)
        x = x1
        out = x if self.return_embeds else self.softmax(x)
        
        return out

class GATN3(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim,
                 dropout, return_embeds=False):
        
        super(GCN, self).__init__()

        self.gcn1 = GATConv(input_dim, hidden_dim)
        self.gcn2 = GATConv(hidden_dim, hidden_dim)
        self.gcn3 = GATConv(hidden_dim, hidden_dim)        
        self.softmax = torch.nn.LogSoftmax()

        self.dropout = dropout
        self.return_embeds = return_embeds
        

    def reset_parameters(self):
        self.gcn1.reset_parameters()
        self.gcn2.reset_parameters()
        self.gcn3.reset_parameters()

    def forward(self, x, edge_index, edge_weight):

        x1 = F.relu(self.gcn1(x, edge_index, edge_weight))
        if self.training:
            x1 = F.dropout(x1, p=self.dropout)

        x1 = F.relu(self.gcn2(x1, edge_index, edge_weight))
        if self.training:
            x1 = F.dropout(x1, p=self.dropout)

        x1 = F.relu(self.gcn3(x1, edge_index, edge_weight))
        if self.training:
            x1 = F.dropout(x1, p=self.dropout)

        x = x1

        out = x if self.return_embeds else self.softmax(x)

        return out
        
class GCN6(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim,
                 dropout, return_embeds=False):
        
        super(GCN, self).__init__()
        
        self.gcn1 = GCNConv(input_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.gcn3 = GCNConv(hidden_dim, output_dim)
        self.gcn4 = GCNConv(hidden_dim, hidden_dim)
        self.gcn5 = GCNConv(hidden_dim, hidden_dim)
        self.gcn6 = GCNConv(hidden_dim, output_dim)         
        self.softmax = torch.nn.LogSoftmax()

        self.dropout = dropout
        self.return_embeds = return_embeds # Skip classification layer and return node embeddings
        

    def reset_parameters(self):
        self.gcn1.reset_parameters()
        self.gcn2.reset_parameters()
        self.gcn3.reset_parameters()
        self.gcn4.reset_parameters()
        self.gcn5.reset_parameters()
        self.gcn6.reset_parameters()

    def forward(self, x, edge_index):

        x1 = F.relu(self.gcn1(x, edge_index))
        if self.training:
            x1 = F.dropout(x1, p=self.dropout)

        x1 = F.relu(self.gcn2(x1, edge_index))
        if self.training:
            x1 = F.dropout(x1, p=self.dropout)

        x1 = F.relu(self.gcn3(x1, edge_index))
        if self.training:
            x1 = F.dropout(x1, p=self.dropout)

        x1 = F.relu(self.gcn4(x1, edge_index))
        if self.training:
            x1 = F.dropout(x1, p=self.dropout)

        x1 = F.relu(self.gcn5(x1, edge_index))
        if self.training:
            x1 = F.dropout(x1, p=self.dropout)

        x1 = F.relu(self.gcn6(x1, edge_index))
        if self.training:
            x1 = F.dropout(x1, p=self.dropout)

        x = x1
        out = x if self.return_embeds else self.softmax(x)

        return out
        
class GCN3(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim,
                 dropout, return_embeds=False):
        
        super(GCN, self).__init__()
        
        self.gcn1 = GCNConv(input_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.gcn3 = GCNConv(hidden_dim, output_dim)        
        self.softmax = torch.nn.LogSoftmax()

        self.dropout = dropout
        self.return_embeds = return_embeds # Skip classification layer and return node embeddings
        

    def reset_parameters(self):
        self.gcn1.reset_parameters()
        self.gcn2.reset_parameters()
        self.gcn3.reset_parameters()

    def forward(self, x, edge_index):

        x1 = F.relu(self.gcn1(x, edge_index))
        if self.training:
            x1 = F.dropout(x1, p=self.dropout)

        x1 = F.relu(self.gcn2(x1, edge_index))
        if self.training:
            x1 = F.dropout(x1, p=self.dropout)

        x1 = F.relu(self.gcn3(x1, edge_index))
        if self.training:
            x1 = F.dropout(x1, p=self.dropout)

        x = x1
        out = x if self.return_embeds else self.softmax(x)

        return out        

class GCN_Graph(torch.nn.Module):
    def __init__(self, CN, input_dim, hidden_dim, output_dim, num_layers, dropout):
        super(GCN_Graph, self).__init__()

        # Choose CN from GATN6, GATN3, GCN6 and GCN3
        self.gnn_node = self.CN(input_dim = input_dim, 
                            hidden_dim = hidden_dim,
                            output_dim = hidden_dim,
                            dropout = dropout,
                            return_embeds=True)

        self.pool = global_mean_pool

        # Output layer
        self.linear = torch.nn.Linear(hidden_dim, output_dim)


    def reset_parameters(self):
      self.gnn_node.reset_parameters()
      self.linear.reset_parameters()

    def forward(self, batched_data):
        #x, edge_index, edge_weight, batch = batched_data.x.reshape([-1, 1]), batched_data.edge_index, batched_data.edge_attr, batched_data.batch
        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch
        #print(x.shape)
        #print(edge_index.shape)
        #print(edge_weight.shape)
        # embed contains the features
        #embed = self.node_encoder(x)
        out = None
        
        embed = self.gnn_node(x, edge_index, edge_attr)
        #print(f'embd: {embed.shape}')
        features = self.pool(embed, batch)
        #print(f'embd: {features.shape}')
        #out = F.relu(self.linear(features))
        out = self.linear(features)
        #out = F.log_softmax(self.linear(features))

        return out    
 
