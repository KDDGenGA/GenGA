import torch
import torch.nn as nn
import torch.nn.functional as F
import time as t
import numpy as np
import math
from tqdm import tqdm
EPS = 1e-15

class Diffusion_Paras():
    '''
    For storing diffusion related parameters.
    '''
    def __init__(self, betas):
        self.betas = betas.detach()
        # define alphas 
        alphas = 1. - betas.detach()
        # alpha^hat
        alphas_cumprod = torch.cumprod(alphas, axis=0).detach()
        # calculations for the forward diffusion q(x_t | x_{t-1})
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).detach()
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod).detach()

        self.sqrt_alphas_cumprod = sqrt_alphas_cumprod.detach()
        self.sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.detach()
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        self.alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        self.posterior_variance = betas * (1. - self.alphas_cumprod_prev) / (1. - alphas_cumprod)

class SinusoidalPositionEmbeddings(nn.Module):
    """
    Takes a tensor of shape (batch_size, 1) as input 
    (i.e. the noise levels of several noisy images in a batch), 
    and turns this into a tensor of shape (batch_size, dim), 
    with dim being the dimensionality of the position embeddings.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        
        embeddings = torch.exp(torch.arange(
            half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        # print(embeddings, embeddings.shape, embeddings.squeeze())
        print(embeddings.squeeze(), embeddings.squeeze().shape)
        return embeddings

def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb

def get_label_embedding(label, embedding_dim):
    """
    generate label embedding
    """
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=label.device)
    emb = label.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb

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

# forward diffusion (using the nice property)
def q_sample(x_start, t, scheduler, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start) # z (it does not depend on t!)
    betas = get_named_beta_schedule(scheduler, 1000)

    if scheduler == 'cosine':
        betas = get_named_beta_schedule('cosine', 1000)
    else:
        betas = get_named_beta_schedule('linear', 1000)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

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


def class_guided_sampling_step(model, clf, data, xt, t, t_next, bs, args, diff_paras, learn_sigma=False, eta=0.0, sampling_type="ddpm", out_x0_t=False):
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
            with torch.enable_grad():
                x_in = xt.detach().requires_grad_(True)
                log_probs = clf(x_in, data.edge_index)
                selected = log_probs[range(len(log_probs)), data.y.view(-1)]
                log_grad = torch.autograd.grad(selected.sum(), x_in)[0]
                scaled_log_grad = log_grad * args.classifier_scale
            xt_next = mean + torch.sqrt(sigma_t) * scaled_log_grad + torch.sqrt(sigma_t) * noise
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

def class_guided_sampling(model, clf, data, diff_paras, args):
    with torch.no_grad():
        x_t = torch.randn_like(data.x, device=data.x.device)
        for i in tqdm(reversed(range(0, args.diffsteps)), desc=f"Generative sampling process", total=args.diffsteps):
            x_t = class_guided_sampling_step(
                model,
                clf,
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


def sampling(model, data, diff_paras, args):
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