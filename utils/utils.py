import numpy as np
import scipy.sparse as sp
import scipy.io as sio
import os
import sys
import torch
import torch.nn as nn
import math
import pickle as pkl
import numpy as np
import random
import torch_geometric
import torch.nn.functional as F
import torch_geometric.datasets as datasets
from torch_geometric.data import Data
import torch_geometric.transforms as transforms
import networkx as nx
from torch_geometric.utils.convert import to_networkx
from torch_geometric.utils import negative_sampling, add_remaining_self_loops, degree, coalesce
from itertools import combinations
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, recall_score, precision_score
import csv
from dgl.data.utils import load_graphs


class Config():
    def __init__(self):
        self.name = "model config"

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
    torch_geometric.seed_everything(seed)

def save_result(fin_result, filename):

    csv_file_name = filename

    column_names = list(fin_result.keys())
    data_rows = [[metric["mean"], metric["std"]] for metric in fin_result.values()]

    # Write the data to the CSV file
    with open(csv_file_name, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Metric", "Mean", "Std"])  # Write header
        for metric, data in zip(column_names, data_rows):
            writer.writerow([metric] + data)

def assign_device(args, block_size : int = 128):

    if args.loc == "Lab":
       device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    else:
       device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
       
    if device != "cpu":
        print("===================================================")
        print(f"===              CUDA cache cleared             ===")
        print(f"===   CUDA max split size has been set to {block_size}   ===")
        print("===================================================")
        torch.cuda.empty_cache()
        #print("CUDA cache cleared.")
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = f"max_split_size_mb:{block_size}"
        #print(f"CUDA max split size has been set to {block_size}.")
    return device

def load_data(args):

    dataset = args.dataset
    s1 = ["BlogCatalog", "ACM", "YelpChi"]
    s2 = ["weibo", "reddit", "Cora"]
    s3 = ['questions', 'tolokers']

    datadir = "./data/"
    if dataset in s1:
        _, attrs, label, adj_label = load_anomaly_detection_dataset(dataset, datadir=datadir)
        adj_sym = torch.LongTensor(adj_label)
        adj_sym = torch.where(adj_sym >=1, torch.ones_like(adj_sym), adj_sym)

        attrs = torch.Tensor(attrs)
        adj_label = sp.coo_matrix(adj_sym)
        indices = np.vstack((adj_label.row, adj_label.col))
        adj_label = torch.LongTensor(indices)
        label = torch.LongTensor(label)
        data = Data(x=attrs, edge_index=adj_label, y=label)

    elif dataset in s2:
        file_path = os.path.join(datadir, dataset + ".pt")
        data = torch.load(file_path)
        if dataset == "Cora":
            data.y = data.y.bool()
        data.y = data.y.long()
    
    elif dataset in s3:
        if args.loc == "nci":
            file_path = os.path.join(datadir, dataset)
        else:
            file_path = os.path.join(datadir, dataset)
        data = from_dgl(file_path)
    return data

def from_dgl(file_path):
    graph = load_graphs(file_path)[0][0]
    src, dst = graph.edges()
    data = Data(
        x = torch.tensor(graph.ndata['feature']),
        edge_index=torch.tensor([src.numpy(), dst.numpy()], dtype=torch.long),
        y = torch.tensor(graph.ndata['label'])
    )
    return data

def gen_train_test_mask(data, train_index, val_index, test_index, args):
    y_bool = data.y.bool()

    data.all_anm = torch.Tensor(data.y==1)
    data.all_norm = torch.Tensor(data.y==0)

    assert data.all_anm.sum() + data.all_norm.sum() == data.x.shape[0]

    train_mask = torch.zeros(data.y.shape[0], dtype=torch.bool, device=args.device)
    train_mask[train_index] = True

    val_mask = torch.zeros_like(train_mask)
    val_mask[val_index] = True

    test_mask = torch.zeros_like(train_mask)
    test_mask[test_index] = True

    data.train_anm = train_mask & y_bool
    data.train_norm = train_mask ^ data.train_anm

    data.val_anm = val_mask & y_bool
    data.val_norm = val_mask ^ data.val_anm

    data.test_anm = test_mask & y_bool
    data.test_norm = test_mask ^ data.test_anm

    data.val_mask = val_mask
    data.test_mask = test_mask
    data.train_mask = data.train_anm | data.train_norm

    if torch.any(data.train_mask & data.test_mask):
        raise ValueError("Train mask and test mask have common elements")

    return data

def load_anomaly_detection_dataset(dataset, datadir="./data/"):
    icdm_datasets = ["BlogCatalog", "ACM"]
    if dataset in icdm_datasets:
        data_mat = sio.loadmat(f'{datadir}/{dataset}.mat')
        adj = data_mat['Network']
        feat = data_mat['Attributes']
        truth = data_mat['Label']
        truth = truth.flatten()
    else:
        data_mat = sio.loadmat(f'{datadir}/YelpChi.mat')
        adj = data_mat['net_rur']
        #adj = data_mat['homo']
        feat = data_mat['features']
        truth = data_mat['label']
        truth = truth.flatten()

    adj_norm = normalize_adj(adj + sp.eye(adj.shape[0]))
    adj_norm = adj_norm.toarray()
    adj = adj + sp.eye(adj.shape[0])
    adj_t = adj.transpose()
    adj_sym = adj + adj_t
    adj_sym = adj_sym.toarray()
    feat = feat.toarray()
    return adj_norm, feat, truth, adj_sym

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()