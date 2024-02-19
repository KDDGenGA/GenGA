import os
import sys
ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)
import logging
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import torch
from torch.optim import Adam
from utils.utils import setup_seed, load_data, gen_train_test_mask, assign_device
from arguments.GenGA_arguments import arg_parser
from models.Detector import Detector
from models.CGenGA import Diffusion_Paras, get_named_beta_schedule
from train import train_noisy_clf
import warnings
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, filename=f"../logs/logs.log", format="%(asctime)s %(name)s %(levelname)s %(message)s", filemode="w")

def setup_logger(args):
    logger = logging.getLogger(f"Noisy_CLF_{args.seed}_{args.dataset}")
    handler = logging.FileHandler(filename=f"../logs/NoisyCLF_log/{args.dataset}/Classifier_seed{args.seed}_{args.dataset}.log", mode="w")
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
    # Gen 5-fold test splits
    skf = StratifiedShuffleSplit(n_splits = args.fold, test_size=1 - args.tr, random_state=1234)
    train_indices, val_indices, test_indices = [], [], []

    split_dir = f"./splits/{args.dataset}"
    if not os.path.exists(split_dir):
        os.makedirs(split_dir)
        for i, spt in enumerate(skf.split(np.zeros(data.x.shape[0]), data.y.numpy())):
            train_index = spt[0]
            test_index = spt[1]
            val_index = test_index[:len(test_index)//2]
            test_index = test_index[len(test_index)//2:]
            train_indices.append(train_index)
            val_indices.append(val_index)
            test_indices.append(test_index)
            np.savetxt(f"{split_dir}/train_fold{i}.txt", train_index.astype(int))
            np.savetxt(f"{split_dir}/val_fold{i}.txt", val_index.astype(int))
            np.savetxt(f"{split_dir}/test_fold{i}.txt", test_index.astype(int))
    else:
        for i in range(args.fold):
            train_indices.append(np.loadtxt(f"{split_dir}/train_fold{i}.txt").astype(int))
            val_indices.append(np.loadtxt(f"{split_dir}/val_fold{i}.txt").astype(int))
            test_indices.append(np.loadtxt(f"{split_dir}/test_fold{i}.txt").astype(int))
    
    for seed in range(0, args.runs):
        args.seed = seed
        setup_seed(seed)
        logger = setup_logger(args)
        logger.info(f" [Arguments]: {args}")
        args.logger = logger
        args.input_dim = data.x.shape[1]
        betas = get_named_beta_schedule(args.scheduler, args.diffsteps)
        diff_paras = Diffusion_Paras(betas)

        n_fold_res_f1 = []
        n_fold_res_pre = []
        n_fold_res_rec = []
        n_fold_res_auc = []
        n_fold_res_aupr = []

        for fold in range(args.fold):
            logger.info("===================================================")
            logger.info(f"========   Start Testing on Fold {fold+1}  ===============")
            logger.info("===================================================")

            data = data.to(device)
            data = gen_train_test_mask(data, train_indices[fold], val_indices[fold], test_indices[fold], args)

            model = Detector(args)
            model = model.to(device)
            logger.info(model)
            
            optimizer = Adam(model.parameters(), lr=5e-3)
            args.curfold = fold
            try:
                val_f1_all, val_pre_all, val_rec_all, val_ap_all, val_auc_all, test_f1_all, test_pre_all, test_rec_all, test_ap_all, test_auc_all = train_noisy_clf(model, data, optimizer, diff_paras, args)
            except Exception as e:
                logger.exception(f"Error: {e}")

            best_epoch = np.argmax(val_auc_all)
            logger.info(f"[Best Epoch: {best_epoch}], Val M-F1: {val_f1_all[best_epoch]:.4f}, Val M-AUC: {val_auc_all[best_epoch]:.4f}, Val M-AUPR: {val_ap_all[best_epoch]:.4f}")
            logger.info(f"[Best Epoch: {best_epoch}], Test M-F1: {test_f1_all[best_epoch]:.4f}, Test AUC: {test_auc_all[best_epoch]:.4f}, Test AUPR: {test_ap_all[best_epoch]:.4f}")

            n_fold_res_f1.append(test_f1_all[best_epoch])
            n_fold_res_pre.append(test_pre_all[best_epoch])
            n_fold_res_rec.append(test_rec_all[best_epoch])
            n_fold_res_auc.append(test_auc_all[best_epoch])
            n_fold_res_aupr.append(test_ap_all[best_epoch])
        
        n_fold_res_f1 = np.array(n_fold_res_f1)
        n_fold_res_pre = np.array(n_fold_res_pre)
        n_fold_res_rec = np.array(n_fold_res_rec)
        n_fold_res_auc = np.array(n_fold_res_auc)
        n_fold_res_aupr = np.array(n_fold_res_aupr)

        logger.info(f'''[Seed {seed} Final Test Result] on {args.dataset}:
            F1: {np.mean(n_fold_res_f1):.4f} +- {np.std(n_fold_res_f1):.2f},
            AUC: {np.mean(n_fold_res_auc):.4f} +- {np.std(n_fold_res_auc):.2f},
            Precision: {np.mean(n_fold_res_pre):.4f} +- {np.std(n_fold_res_pre):.2f},
            Recall: {np.mean(n_fold_res_rec):.4f} +- {np.std(n_fold_res_rec):.2f},
            AUPR: {np.mean(n_fold_res_aupr):.4f} +- {np.std(n_fold_res_aupr):.2f}''')

        if np.mean(n_fold_res_auc) > best_result:
            best_result = np.mean(n_fold_res_auc)
            model.save_model(args, f"{args.dataset}_NoisyCLF")