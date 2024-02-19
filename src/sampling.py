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
from utils.diff_utils import class_guided_sampling, sampling
from arguments.GenGA_arguments import arg_parser
from models.CGenGA import Diffusion_Paras, get_named_beta_schedule, CGenGA, q_sample
from models.Detector import Detector
from early_stop import EarlyStopping, Stop_args

import warnings
warnings.filterwarnings("ignore")

def setup_logger(args):
    if not os.path.exists(f"../logs/Sampling_Log/"):
        os.mkdir(f"../logs/Sampling_Log/")

    if args.class_cond:
        logger = logging.getLogger(f"Sampling_{args.dataset}_{args.loss_type}_{args.cur_f}")
        handler = logging.FileHandler(filename=f"../logs/Sampling_Log/Sampling_{args.dataset}_{args.loss_type}_{args.cur_f}.log", mode="w")
    else:
        logger = logging.getLogger(f"Sampling_{args.dataset}_{args.loss_type}_Unguide")
        handler = logging.FileHandler(filename=f"../logs/Sampling_Log/Sampling_{args.dataset}_{args.loss_type}_Unguide.log", mode="w")

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

def save_gen(aug_g, name, args):
    path = f"../generated/{args.dataset}/"
    if not os.path.exists(path):
        os.makedirs(path)

    torch.save(aug_g, os.path.join(path, f"{name}_S{args.classifier_scale}.pth"))

def class_uncond_sampling(data, args):

    betas = get_named_beta_schedule(args.scheduler, args.diffsteps)
    diff_paras = Diffusion_Paras(betas)

    logger = setup_logger(args)
    logger.info(f" [Arguments]: {args}")
    args.logger = logger
    logger.info("===================================================")
    logger.info(f"========  Unconditional Sampling on {args.dataset}  ============")
    logger.info("===================================================")

    data = data.to(device)

    diff_model = CGenGA(args)
    diff_model = diff_model.to(device)
    diff_model.load_model(args, name=f"CGenGA_{args.dataset}_{args.loss_type}_Uncond")

    if args.class_guide:
        classifier = Detector(args)
        classifier = classifier.to(device)
        classifier.load_model(args, name=f"{args.dataset}_NoisyCLF")
        classifier.eval()

    for i in range(args.gs):
        if args.class_guide:
            aug_graph = class_guided_sampling(diff_model, classifier, data, diff_paras, args)
            name = f"{args.dataset}_UC_CG_{i}"
        else:
            aug_graph = sampling(diff_model, data, diff_paras, args)
            name = f"{args.dataset}_UC_NCG_{i}"
        if args.saveg == "yes":
            save_gen(F.normalize(aug_graph, dim=1), name, args)

    logger.info("Sampling Done!")
    pass

if __name__ == "__main__":
    
    best_result = 0
    args = arg_parser()
    device = assign_device()
    args.device = device

    data = load_data(args)
    args.input_dim = data.x.shape[1]
    
    if not args.class_cond:
        class_uncond_sampling(data, args)
