import numpy as np
import networkx as nx
from networkx.algorithms import centrality as cen
from networkx.algorithms import efficiency_measures as eff
from torch_geometric.utils import remove_self_loops
from torch_sparse import coalesce
from torch_geometric.data import Dataset, Data
import torch
import torch.nn.functional as F
import pandas as pd
import os
from tqdm import tqdm

#Convert vectorized correlatiom matrix to symmetric matrix
def read_cor(filename, n_ICs):
    IC = np.loadtxt(filename)
    #values, counts = np.unique(IC, return_counts=True)
    #IC_nodiag = np.delete(IC, np.where(IC == values[np.argmax(counts)]))
    size = n_ICs 
    corr_matrix = np.zeros((size,size))
    corr_matrix[np.tril_indices(corr_matrix.shape[0], k = -1)] = IC
    corr_matrix = corr_matrix + corr_matrix.T
    return corr_matrix
  
#Only keep edges with positive correlation
def pos_net(correltation_matrix):
  correltation_matrix[correltation_matrix < 0] = 0
  return correltation_matrix

#Convert correlation matrix to adjacency matrix in networkx format
def adj_netx(correltation_matrix):
  return nx.from_numpy_array(correltation_matrix)

#Build PyG Dataset
class MriDataset(Dataset):
  def __init__(self, root, filename, test=False, transform=None, pre_transform=None):
  #def __init__(self, root, filename, node_feature, test=False, transform=None, pre_transform=None):
      self.test = test
      self.filename = filename
      #self.node_feature = node_feature
      super(MriDataset, self).__init__(root, transform, pre_transform)

  @property
  def raw_file_names(self):
      return self.filename

  @property
  def processed_file_names(self):
      self.data = pd.read_csv(self.raw_paths[0]).reset_index()

      if self.test:
          return [f'data_test_{i}.pt' for i in list(self.data.index)]
      else:
          return [f'data_{i}.pt' for i in list(self.data.index)]

  def download(self):
      pass

  def process(self):
      self.data = pd.read_csv(self.raw_paths[0])

      for index, mri in tqdm(self.data.iterrows(), total=self.data.shape[0]):
          mri_net = adj_netx(pos_net(read_cor(mri['txt_files'], 25)))

          # Get node features
          #if (self.node_feature == 'effdeg'):
          node_feats = torch.cat((self._get_node_features_eff(mri_net), self._get_node_features_degree(mri_net)), -1)
          #else if (self.node_feature == 'eff'):
          #  node_feats = self._get_node_features_eff(mri_net)
          #else if (self.node_feature == 'deg'):
          #  node_feats = self._get_node_features_degree(mri_net)
          #else:
          #  raise ValueError('Please choose node features from eff, deg and effdeg.')
          # Get edge features
          edge_index, edge_feats = self._get_edge(mri_net)
          #Get labels
          label = self._get_labels(mri['autism'])
          #print(label)

          # Create data object
          data = Data(x=node_feats, 
                      edge_index=edge_index,
                      edge_attr=edge_feats,
                      y=label
                      ) 
          #print(data)
          if self.test:
              torch.save(data, 
                  os.path.join(self.processed_dir, 
                                f'data_test_{index}.pt'))
          else:
              torch.save(data, 
                  os.path.join(self.processed_dir, 
                                f'data_{index}.pt'))
              
  def _get_node_features_eff(self, mri_net):
      all_node_feats = []
      lengths = nx.all_pairs_bellman_ford_path_length(mri_net)
      for source, targets in lengths:
          g_eff = 0
          for target, distance in targets.items():
              if distance > 0:
                  g_eff += distance
          all_node_feats.append(g_eff)

      all_node_feats = np.asarray(all_node_feats)
      return torch.tensor(all_node_feats, dtype=torch.float).reshape([-1, 1])

  def _get_node_features_degree(self, mri_net):

      all_node_feats = np.array(mri_net.degree(weight = 'weight'))[:,1]
      return torch.tensor(all_node_feats, dtype=torch.float).reshape([-1, 1])

  def _get_edge(self, mri_net):
      all_edge_feats = []

      A = nx.to_scipy_sparse_matrix(mri_net)
      adj = A.tocoo()
      all_edge_feats = np.zeros(len(adj.row))
      for i in range(len(adj.row)):
          #all_edge_feats[i] = t1[adj.row[i], adj.col[i]]
          all_edge_feats[i] = nx.to_numpy_array(mri_net)[adj.row[i], adj.col[i]]
      all_edge_feats = np.asarray(all_edge_feats)

      edge_index = np.stack([adj.row, adj.col])
      edge_index, edge_att = remove_self_loops(torch.from_numpy(edge_index), torch.from_numpy(all_edge_feats))
      edge_index = edge_index.long()        
      #return coalesce(edge_index, all_edge_feats, 25, 25)
      return torch.tensor(edge_index, dtype=torch.long), torch.tensor(edge_att, dtype=torch.float64)
  def _get_labels(self, label):
      label = np.asarray([label])
      #print(label)
      #print(torch.tensor(label, dtype=torch.int64))
      return torch.tensor(label, dtype=torch.int64)

  def len(self):
      return self.data.shape[0]

  def get(self, idx):
      data = torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'))   
      return data
