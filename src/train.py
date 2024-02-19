import torch
import torch.nn.functional as F
from early_stop import EarlyStopping, Stop_args_clf
from models.CGenGA import q_sample
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score, roc_auc_score
import logging

def train_noisy_clf(model, data, optimizer, diff_paras, args):

    stopping_args = Stop_args_clf(patience=args.patience, max_epochs=args.clfepochs)
    early_stopping = EarlyStopping(model, **stopping_args)
    logger = args.logger

    val_f1_all = []
    val_pre_all = []
    val_rec_all = []
    val_ap_all = []
    val_auc_all = []

    test_f1_all = []
    test_pre_all = []
    test_rec_all = []
    test_ap_all = []
    test_auc_all = []

    for epoch in range(1, 1 + args.clfepochs):
        model.train()
        t = torch.randint(0, args.diffsteps, (1,), device=args.device).long()
        noise = torch.randn_like(data.x, device=t.device)
        x_t = q_sample(data, t=t, noise=noise, diff_paras = diff_paras)
        x_t = x_t.detach()

        preds, train_loss = model.fit(x_t, data, optimizer)

        val_loss = F.nll_loss(preds[data.val_mask], data.y[data.val_mask])
        test_loss = F.nll_loss(preds[data.test_mask], data.y[data.test_mask])
        
        train_f1 = f1_score(data.y[data.train_mask].cpu(), preds[data.train_mask].argmax(dim=-1).cpu(), average='macro')
        train_pre = precision_score(data.y[data.train_mask].cpu(), preds[data.train_mask].argmax(dim=-1).cpu(), average='macro')
        train_rec = recall_score(data.y[data.train_mask].cpu(), preds[data.train_mask].argmax(dim=-1).cpu(), average='macro')
        train_ap = average_precision_score(data.y[data.train_mask].cpu(), preds[data.train_mask].argmax(dim=-1).cpu())
        train_auc = roc_auc_score(data.y[data.train_mask].cpu(), preds[data.train_mask].argmax(dim=-1).cpu())

        val_f1 = f1_score(data.y[data.val_mask].cpu(), preds[data.val_mask].argmax(dim=-1).cpu(), average='macro')
        val_pre = precision_score(data.y[data.val_mask].cpu(), preds[data.val_mask].argmax(dim=-1).cpu(), average='macro')
        val_rec = recall_score(data.y[data.val_mask].cpu(), preds[data.val_mask].argmax(dim=-1).cpu(), average='macro')
        val_ap = average_precision_score(data.y[data.val_mask].cpu(), preds[data.val_mask].argmax(dim=-1).cpu())
        val_auc = roc_auc_score(data.y[data.val_mask].cpu(), preds[data.val_mask].argmax(dim=-1).cpu())

        test_f1 = f1_score(data.y[data.test_mask].cpu(), preds[data.test_mask].argmax(dim=-1).cpu(), average='macro')
        test_pre = precision_score(data.y[data.test_mask].cpu(), preds[data.test_mask].argmax(dim=-1).cpu(), average='macro')
        test_rec = recall_score(data.y[data.test_mask].cpu(), preds[data.test_mask].argmax(dim=-1).cpu(), average='macro')
        test_ap = average_precision_score(data.y[data.test_mask].cpu(), preds[data.test_mask].argmax(dim=-1).cpu())
        test_auc = roc_auc_score(data.y[data.test_mask].cpu(), preds[data.test_mask].argmax(dim=-1).cpu())

        if epoch % 50 == 0:
            logger.info(f"Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Train M-F1: {train_f1:.4f}, Train M-Pre: {train_pre:.4f}, Train M-Rec: {train_rec:.4f}, Train M-AP: {train_ap:.4f}, Train M-AUC: {train_auc:.4f}")

            logger.info(f"Epoch: {epoch:03d}, Val Loss: {val_loss:.4f}, Val M-F1: {val_f1:.4f}, Val M-Pre: {val_pre:.4f}, Val M-Rec: {val_rec:.4f}, Val M-AP: {val_ap:.4f}, Val M-AUC: {val_auc:.4f}")

            logger.info(f"Epoch: {epoch:03d}, Test Loss: {test_loss:.4f}, Test M-F1: {test_f1:.4f}, Test M-Pre: {test_pre:.4f}, Test M-Rec: {test_rec:.4f}, Test M-AP: {test_ap:.4f}, Test M-AUC: {test_auc:.4f}")

        val_f1_all.append(val_f1)
        val_pre_all.append(val_pre)
        val_rec_all.append(val_rec)
        val_ap_all.append(val_ap)
        val_auc_all.append(val_auc)

        test_f1_all.append(test_f1)
        test_pre_all.append(test_pre)
        test_rec_all.append(test_rec)
        test_ap_all.append(test_ap)
        test_auc_all.append(test_auc)

        if early_stopping.check([val_auc], epoch):
            break
        
    return val_f1_all, val_pre_all, val_rec_all, val_ap_all, val_auc_all, test_f1_all, test_pre_all, test_rec_all, test_ap_all, test_auc_all
