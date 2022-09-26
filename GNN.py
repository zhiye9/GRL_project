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
