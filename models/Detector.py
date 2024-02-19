import torch
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv
import torch.nn as nn
import torch.nn.functional as F
import os

class Detector(nn.Module):
    def __init__(self, args):
        super(Detector, self).__init__()
        self.name = "Anomaly Detector"
        if args.clf == "GCN":
            self.clf = nn.ModuleList([
                GCNConv(args.input_dim, args.clf_dim),
                GCNConv(args.clf_dim, args.clf_dim),
                nn.Linear(args.clf_dim, 2),
            ])
        elif args.clf == "GAT":
            self.clf = nn.ModuleList([
                GATConv(args.input_dim, args.clf_dim, heads=args.heads),
                GATConv(args.clf_dim * args.heads,args.clf_dim),
                nn.Linear(args.clf_dim * args.heads,2),
            ])
        else:
            raise NotImplementedError("Not Implemented Anomaly Detector")
    
    def forward(self, x, edge_index, args = None):
        preds = self.clf[0](x, edge_index).relu()
        preds = self.clf[1](preds, edge_index).relu()
        preds = F.log_softmax(self.clf[2](preds), dim=-1)
        return preds
    
    def fit(self, x, data, optimizer):
        self.clf.train()
        optimizer.zero_grad()
        preds = self(data.x, data.edge_index)
        loss = F.nll_loss(preds[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        return preds, loss.item()

    def test(self, data):
        self.clf.eval()
        with torch.no_grad():
            preds = self.clf(data.x, data.edge_index)
            val_loss = F.nll_loss(preds[data.val_mask], data.y[data.val_mask])
            test_loss = F.nll_loss(preds[data.test_mask], data.y[data.test_mask])
        return preds, val_loss.item(), test_loss.item()
    
    def save_model(self, args, name=None):
        path = f"../saved_models/{args.dataset}"
        if not os.path.exists(path):
            os.makedirs(path)

        if name is not None:
            torch.save(self.state_dict(), os.path.join(path, f"{name}.pth"))
        else:
            torch.save(self.state_dict(), os.path.join(path, f"{args.clf}_Detector.pth"))
    
    def load_model(self, args, name=None):
        if name is not None:
            path = os.path.join(f"../saved_models/{args.dataset}/{name}.pth")
        else:
            path = os.path.join(f"../saved_models/{args.dataset}_Detector.pth")
        self.load_state_dict(torch.load(path))