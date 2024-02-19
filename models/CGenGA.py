import torch
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv
import torch.nn as nn
from torch.nn import ModuleList
import torch.nn.functional as F
import time as t
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, roc_curve
import math
import os
from tqdm import tqdm
from utils.utils import AER
from utils.diff_utils import get_timestep_embedding, get_label_embedding, Diffusion_Paras, SinusoidalPositionEmbeddings
from models.Detector import Detector
EPS = 1e-15

def nonlinearity(x):
    return x * torch.sigmoid(x)

class Noisy_CLF(nn.Module):
    def __init__(self, args):
        super(Noisy_CLF, self).__init__()
        self.name = "Noisy Classifier"

        if args.noisy_clf == "GCN":
            self.noisy_clf = nn.ModuleList([
                GCNConv(args.input_dim, args.clf_dim),
                GCNConv(args.clf_dim, args.clf_dim),
                nn.Linear(args.clf_dim,2),
            ])
        elif args.noisy_clf == "GAT":
            self.noisy_clf = nn.ModuleList([
                GATConv(args.input_dim, 128),
                GATConv(128,64),
                nn.Linear(64,2),
            ])
        else:
            raise NotImplementedError("Not Implemented Noisy CLF")

    def forward(self, x, edge_index, args = None):
        preds = self.noisy_clf[0](x, edge_index).relu()
        preds = self.noisy_clf[1](preds, edge_index).relu()
        preds = F.softmax(self.noisy_clf[2](preds), dim=-1)
        return preds

class CGenGA(nn.Module):
    def __init__(self, args):
        super(CGenGA, self).__init__()
        self.name = "Classifier guided GenGA"
        self.downGNN = ModuleList()
        self.upGNN = ModuleList()
        self.act = nn.SiLU()

        if args.gnn == "GCN":
            self.downGNN.append(
                GCNConv(args.input_dim, args.gnn_dim)
            )
            self.downGNN.append(
                GCNConv(args.gnn_dim, args.gnn_dim//2)
            )
            self.upGNN.append(
                GCNConv(args.gnn_dim//2, args.gnn_dim),
            )
            self.upGNN.append(
                GCNConv(2 * args.gnn_dim, args.input_dim),
            )
        else:
            raise NotImplementedError("Not Implemented GNN Layers")
        
        if args.with_time_emb:
            time_dim = args.input_dim
            self.time_mlp = nn.ModuleList([
                nn.Linear(args.input_dim, time_dim),
                nn.Linear(time_dim, time_dim),
            ])
        else:
            time_dim = None
            self.time_mlp = None

        if args.class_cond:
            self.label_emb = nn.Embedding(2, args.input_dim)
            self.label_emb_m = 0
        else:
            self.label_emb_m = None

        self.residual_links = []

    def init_lable_emb(self, data):
        self.label_emb_m = torch.zeros_like(data.x).to(data.x.device)
        self.label_emb_m[data.train_anm] = self.label_emb(torch.tensor([1], device=data.x.device))
        self.label_emb_m[data.train_norm] = self.label_emb(torch.tensor([0], device=data.x.device))

    def save_model(self, args, name="Gdiff"):
        path = f"../saved_models/{args.dataset}"
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.state_dict(), os.path.join(path, f"{name}.pth"))

    def load_model(self, args, name="Gdiff"):
        path = os.path.join(f"../saved_models/{args.dataset}/{name}.pth")
        self.load_state_dict(torch.load(path))
        
    def load_weights(self, args):
        self.noisy_clf.load_model(args, f"{args.dataset}_NoisyCLF")

    def forward(self, data, noise_graph_X_t, t, diff_paras, args, stage="diffusion", aug_graphs=None):
        results = []
        if stage == "diffusion":
            pred_noise = self.diffusion_step(data, noise_graph_X_t, t, args)
            return pred_noise
        return results
    
    def diffusion_step(self, data, noise_graph_X_t, t, args):
        # add t embedding to data x
        time_emb = get_timestep_embedding(t, data.x.shape[1]).detach()
        time_emb = self.time_mlp[0](time_emb)
        time_emb = nonlinearity(time_emb)
        time_emb = self.time_mlp[1](time_emb)
        x_t = noise_graph_X_t + time_emb

        if args.class_cond:
            self.init_lable_emb(data)
            x_t = x_t + self.label_emb_m
        
        for i, gnn in enumerate(self.downGNN):
            if i == 0:
                pred_noise = gnn(x_t, data.edge_index)#.relu()
                pred_noise = self.act(pred_noise)
                self.residual_links.append(pred_noise)
            else:
                pred_noise = gnn(pred_noise, data.edge_index)#.relu()
                pred_noise = self.act(pred_noise)

        for i, gnn in enumerate(self.upGNN):
            if i == 0:
                pred_noise = gnn(pred_noise, data.edge_index)#.relu()
                pred_noise = self.act(pred_noise)
            else:
                pred_noise = gnn(torch.cat([pred_noise,self.residual_links.pop()],dim=1), data.edge_index)
        pred_noise = nonlinearity(pred_noise)
        return pred_noise

# Scheduler
def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return torch.linspace(
            beta_start, beta_end, num_diffusion_timesteps
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    
    return torch.tensor(betas)

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def q_sample(data, t, noise, diff_paras):
    if noise is None:
        noise = torch.randn_like(data.x).to(t.device) # z (it does not depend on t!)

    sqrt_alphas_cumprod_t = extract(diff_paras.sqrt_alphas_cumprod, t, data.x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        diff_paras.sqrt_one_minus_alphas_cumprod, t, data.x.shape
    )
    return sqrt_alphas_cumprod_t.to(t.device) * data.x + sqrt_one_minus_alphas_cumprod_t.to(t.device) * noise

def Sampling_step(model, data, xt, t, t_next, bs, args, diff_paras, learn_sigma=False, eta=0.0, sampling_type="ddpm", out_x0_t=False):
    et = model.diffusion_step(data, xt, t, args)
    if learn_sigma:
        et, sigma_t = torch.split(et, et.shape[1]//2, dim=1)
    else:
        sigma_t = extract(diff_paras.posterior_variance, t, xt.shape)
    bt = extract(bs, t, xt.shape)
    at_bar = extract((1.0 - bs).cumprod(dim=0), t, xt.shape)
    xt_next = torch.zeros_like(xt)
    
    if t_next.sum() == -t_next.shape[0]:
        at_next = torch.ones_like(at_bar)
    else:
        at_next = extract((1.0 - bs).cumprod(dim=0), t_next, xt.shape)

    if sampling_type == "ddpm":
        # Eq.(124) on P15
        weight = bt/torch.sqrt(1-at_bar)
        mean = (xt - weight * et) / torch.sqrt(1.0 - bt)
        noise = torch.randn_like(xt)
        if t == 0:
            xt_next = mean
        else:
            xt_next = mean + torch.sqrt(sigma_t) * noise
        xt_next = xt_next.float()

    if sampling_type == "ddim":
        x0_t = (xt - et * (1 - at_bar).sqrt()) / at_bar.sqrt()
        if eta == 0:
            xt_next = at_next.sqrt() * x0_t + (1 - at_next).sqrt() * et
        elif at_bar > (at_next):
            print('Inversion process is only possible with eta = 0')
            raise ValueError
        else:
            c1 = eta * ((1 - at_bar / (at_next)) * (1 - at_next) / (1 - at_bar)).sqrt()
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c2 * et + c1 * torch.randn_like(xt)
    if out_x0_t == True:
        return xt_next, x0_t
    else:
        return xt_next

def Sampling(model, data, diff_paras, args):
    with torch.no_grad():
        x_t = torch.randn_like(data.x, device=data.x.device)
        for i in tqdm(reversed(range(0, args.diffsteps)), desc=f"Generative sampling process", total=args.diffsteps):
            x_t = Sampling_step(
                model,
                data,
                x_t,
                torch.tensor([i], device=data.x.device), # time
                torch.tensor([i-1], device=data.x.device),
                diff_paras.betas,
                args,
                diff_paras,
                learn_sigma=False, eta=0.0, sampling_type="ddpm", out_x0_t=False
            )
    return x_t



def save_gen(aug_g, i, args):
    torch.save(aug_g, f"../generated/{args.dataset}_F{args.curfold}.pt")
