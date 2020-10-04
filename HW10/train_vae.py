import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


from utils import *
from model import *
from dataset import *

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
seed = 1114
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

if sys.argv[2] in ["fvae", "FVAE"]:
    case = "fvae"
elif sys.argv[2] in ["cvae", "CVAE"]:
    case = "cvae"
else:
    print("Wrong model structure name!")
    exit()

BATCH = 128
LR = 1e-3
EPOCH = 450

def criterion(reconst_x, x, mu, logvar):
    REC = F.mse_loss(reconst_x, x)
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    #REC = F.mse_loss(reconst_x, x, reduction="sum")
    #KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return REC + KLD, REC, KLD

if __name__ == "__main__":
    train_x = np.load(sys.argv[1], allow_pickle=True)
    train_x = preprocess(train_x)
    if case == "fvae":
        train_x = train_x.reshape(train_x.shape[0], -1)
    train_dataset = ImageDataset(train_x)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True)

    if sys.argv[4] == '1':
        val_x = np.load(sys.argv[5], allow_pickle=True)
        val_x = preprocess(val_x)
        val_y = np.load(sys.argv[6], allow_pickle=True)
        if case == "fvae":
            val_x = val_x.reshape(val_x.shape[0], -1)
        val_dataset = ImageDataset(val_x)
        val_dataloader = DataLoader(val_dataset, batch_size=BATCH)
    
    if case == "fvae":
        model = FVAE().cuda()
    elif case == "cvae":
        model = CVAE().cuda()
    total_param = sum(p.numel() for p in model.parameters())
    trainable_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("{} parameters with {} trainable".format(total_param, trainable_param), flush=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    for epoch in range(EPOCH):
        model.train()
        train_loss = [0, 0, 0]
        for data in train_dataloader:
            x = data.cuda()

            (mu, logvar, _), reconst_x = model(x)
            loss, rec_loss, kld_loss = criterion(reconst_x, x, mu, logvar)
            train_loss = [train_loss[0] + loss, train_loss[1] + rec_loss, train_loss[2] + kld_loss]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("Epoch {}/{}: Train | loss = {:.5f} REC = {:.5f} KLD = {:.5f}".format(epoch + 1, EPOCH, train_loss[0] / len(train_dataloader), train_loss[1] / len(train_dataloader), train_loss[2] / len(train_dataloader)), flush=True)

        if (epoch + 1) % 10 == 0:
            if sys.argv[4] == '1':
                pred_x = []
                pred_mu = []
                pred_logvar = []
                model.eval()
                for data in val_dataloader:
                    x = data.cuda()
                    (mu, logvar, _), reconst_x = model(x)
                    pred_x.append(reconst_x.detach().cpu().numpy())
                    pred_mu.append(mu.detach().cpu().numpy())
                    pred_logvar.append(logvar.detach().cpu().numpy())
                pred_x = np.concatenate(pred_x, axis=0)
                pred_mu = np.concatenate(pred_mu, axis=0)
                pred_logvar = np.concatenate(pred_logvar, axis=0)
                anomaly = np.mean(np.square(val_x - pred_x).reshape(val_x.shape[0], -1), axis=1) - 0.5 * np.mean(1 + pred_logvar - pred_mu ** 2 - np.exp(pred_logvar), axis=1)
                print(np.mean(anomaly))
                print("Val | ROCAUC = {:.5f}\n".format(roc_auc_score(val_y, anomaly, average="micro")), flush=True)

    print("Saving checkpoint...\n", flush=True)
    torch.save(model.state_dict(), sys.argv[3])
