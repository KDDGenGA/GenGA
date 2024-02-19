import os
import sys
ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)
import logging
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import torch
from torch.optim import Adam
import torch.nn.functional as F
from utils.utils import setup_seed, load_data, gen_train_test_mask, assign_device
from arguments.GenGA_arguments import arg_parser
from models.CGenGA import Diffusion_Paras, get_named_beta_schedule, CGenGA, q_sample
from early_stop import EarlyStopping, Stop_args

import warnings
warnings.filterwarnings("ignore")

def setup_logger(args):
    logger = logging.getLogger(f"Diffusion_{args.dataset}_{args.loss_type}_{args.cur_f}")
    handler = logging.FileHandler(filename=f"../logs/Diffusion_Log/Diffusion_{args.dataset}_{args.loss_type}_{args.cur_f}.log", mode="w")
    formatter = logging.Formatter('%(asctime)s %(name)s %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

def assign_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device != "cpu":
        import os
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
        print("Set CUDA Block Size to 128")
    return device

if __name__ == "__main__":
    
    best_result = 0
    args = arg_parser()
    device = assign_device()
    args.device = device

    data = load_data(args)    
    args.input_dim = data.x.shape[1]
    
    train_indices, val_indices, test_indices = [], [], []
    split_dir = f"./splits/{args.dataset}"
    for i in range(args.fold):
        train_indices.append(np.loadtxt(f"{split_dir}/train_fold{i}.txt").astype(int))
        val_indices.append(np.loadtxt(f"{split_dir}/val_fold{i}.txt").astype(int))
        test_indices.append(np.loadtxt(f"{split_dir}/test_fold{i}.txt").astype(int))

    for fold in range(args.fold):
        args.cur_f = fold
        logger = setup_logger(args)

        logger.info(f" [Arguments]: {args}")
        args.logger = logger

        logger.info("===================================================")
        logger.info(f"========   Graph Diffusion on Fold {fold}  ===============")
        logger.info("===================================================")

        data = data.to(device)
        data = gen_train_test_mask(data, train_indices[fold], val_indices[fold], test_indices[fold], args)

        model = CGenGA(args)
        model = model.to(device)

        optimizer = Adam(model.parameters(), lr=5e-3)
        
        betas = get_named_beta_schedule(args.scheduler, args.diffsteps)
        diff_paras = Diffusion_Paras(betas)
        stopping_args = Stop_args(patience=args.patience, max_epochs=args.diffsteps)
        early_stopping = EarlyStopping(model, **stopping_args)
        best_loss = 10

        for epoch in range(1, args.diff_epochs):
            model.train()
            optimizer.zero_grad()

            t = torch.randint(0, args.diffsteps, (1,), device=args.device).long()
            noise = torch.randn_like(data.x, device=t.device)
            noise_graph_X_t = q_sample(data, t=t, noise=noise, diff_paras=diff_paras)
            noise_graph_X_t = noise_graph_X_t.detach()

            predicted_noise = model(data, noise_graph_X_t, t, diff_paras, args, stage="diffusion")
            
            if args.loss_type == 'l1':
                diff_loss = F.l1_loss(noise, predicted_noise)
            elif args.loss_type == 'l2':
                diff_loss = F.mse_loss(noise, predicted_noise)
            elif args.loss_type == 'huber':
                diff_loss = F.smooth_l1_loss(noise, predicted_noise)
            else:
                raise NotImplementedError()
            
            diff_loss.backward()
            optimizer.step()

            if epoch % 10 == 0:
                logger.info(f" [Epoch {epoch}] Loss: {diff_loss.item()}")
            
            if diff_loss.item() < best_loss:
                best_loss = diff_loss.item()
                model.save_model(args, name=f"CGenGA_{args.dataset}_{args.loss_type}_{args.cur_f}")

            if early_stopping.check([diff_loss.item()], epoch):
                break
