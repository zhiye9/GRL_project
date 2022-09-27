from ogb.graphproppred import Evaluator
from torch_geometric.data import DataLoader
from tqdm.notebook import tqdm
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
from sklearn.svm import SVC
from tqdm import tqdm
from torch_geometric.nn import global_add_pool, global_mean_pool
from pyg_dataset import MriDataset
from GNN import *

datasets = MriDataset(root = "data/", filename = "df_files.csv")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device: {}'.format(device))

def train_val_test_split(kfold = 5, fold = 0, n_sub):
    label = pd.read_csv('./data/raw/df_files.csv')['autism']
    id = list(range(n_sub))


    import random
    random.seed(123)
    random.shuffle(id)

    kf = StratifiedKFold(n_splits=kfold, random_state=123,shuffle = True)
    kf2 = StratifiedKFold(n_splits=kfold-1, shuffle=True, random_state = 666)


    test_index = list()
    train_index = list()
    val_index = list()

    for tr,te in kf.split(np.array(id), label):
        test_index.append(te)
        tr_id, val_id = list(kf2.split(tr, label[tr]))[0]
        train_index.append(tr[tr_id])
        val_index.append(tr[val_id])

    train_id = train_index[fold]
    test_id = test_index[fold]
    val_id = val_index[fold]

    return train_id,val_id,test_id

tr_index,val_index,te_index = train_val_test_split(fold=0, n_sub = 80)

train_loader = DataLoader(dataset[tr_index], batch_size=16, shuffle=True, num_workers=0)
valid_loader = DataLoader(dataset[val_index], batch_size=16, shuffle=False, num_workers=0)
test_loader = DataLoader(dataset[te_index], batch_size=16, shuffle=False, num_workers=0)

def train(model, device, data_loader, optimizer, loss_fn):
    # This function trains the GNN for graph classification
    model.train()
    loss = 0
    
    # recall, that tqdm just makes nice progression bars during the training
    for step, batch in enumerate(tqdm(data_loader, desc="Iteration")):
      #print(data_loader)
      #print(f'step_train: {step}')
      batch = batch.to(device)
      #print(batch)

      if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
          print("batch = 0")
      else:
        ## ignore nan targets (unlabeled) when computing training loss.
        is_labeled = batch.y == batch.y

        optimizer.zero_grad()
        out = model(batch)
        #out = model(batch)[:,0].reshape([-1, 1])
        #print(f'out: {out}')
        #print(f'y: {batch.y}')
        loss = loss_fn(out[is_labeled], batch.y[is_labeled].float().reshape([-1, 1]))
        #loss = loss_fn(out[is_labeled], torch.nn.functional.one_hot(batch.y)[is_labeled].float())
        loss.backward()
        optimizer.step()

    return loss.item()
  
def eval(model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred = model(batch)

            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)
        
 args = {
    'device': device,
    'num_layers': 8,
    'hidden_dim': 16,
    'dropout': 0.10,
    'lr': 0.0001,
    'epochs': 300,
}

model = GCN_Graph(GATN6, 2, 16, 1, args['num_layers'],
            args['dropout']).to(device)          
evaluator = Evaluator(name='ogbg-molhiv')

import copy

model.to(device).reset_parameters()

optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
loss_fn = torch.nn.BCEWithLogitsLoss()

train_rocauc_plot = []
valid_rocauc_plot = []
test_rocauc_plot = []
loss_BCE = []
y_pre = []
y_tru = []

best_model = None
best_valid_rocauc = 0
for epoch in range(1, 1 + args["epochs"]):
  print('Training...')
  loss_rocauc = train(model, device, train_loader, optimizer, loss_fn)

  print('Evaluating...')
  train_result = eval(model, device, train_loader, evaluator)
  val_result = eval(model, device, valid_loader, evaluator)
  test_result = eval(model, device, test_loader, evaluator)

  train_rocauc, valid_rocauc, test_rocauc = train_result['rocauc'], val_result['rocauc'], test_result['rocauc']
  if valid_rocauc > best_valid_rocauc:
      best_valid_rocauc = valid_rocauc
      best_model = copy.deepcopy(model)
  print(f'Epoch: {epoch:02d}, '
        f'Loss: {loss_rocauc:.4f}, '
        f'Train: {100 * train_rocauc:.6f}%, '
        f'Valid: {100 * valid_rocauc:.6f}% '
        f'Test: {100 * test_rocauc:.6f}%')


#Train baseline model
df_files = pd.read_csv('../df_age21_33.csv')
df_files['txt'] = df_files['eid'].astype(str) + '.txt'

df_data = []
for i in range(0, df_files.shape[0]):
    tem = np.loadtxt(df_files['txt'].loc[i])
    df_data.append(tem)
    print("\r Process{}%".format(round((i+1)*100/df_files.shape[0])), end="")

train_data = np.array([df_data[i] for i in tr_index])
valid_data = np.array([df_data[i] for i in val_index])
test_data = np.array([df_data[i] for i in te_index])

y = np.array(df_files['autism'])

train_y = y[tr_index]
valid_y = y[val_index]
test_y = y[te_index]

model1 =  SVC(kernel = 'linear',  probability = True, C = 0.1)
model2 =  SVC(kernel = 'linear',  probability = True, C = 0.01)
model3 =  SVC(kernel = 'linear',  probability = True, C = 0.001)

model1.fit(train_data, train_y)
model2.fit(train_data, train_y)
model3.fit(train_data, train_y)

y_pred1 = model1.predict_proba(valid_data)[:, 1]
y_pred2 = model2.predict_proba(valid_data)[:, 1]
y_pred3 = model3.predict_proba(valid_data)[:, 1]

roc1 = roc_auc_score(valid_y, y_pred1)
roc2 = roc_auc_score(valid_y, y_pred2)
roc3 = roc_auc_score(valid_y, y_pred3)
max(roc1, roc2, roc3)

roc = roc_auc_score(test_y, y_pred)
