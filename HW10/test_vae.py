import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
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

if sys.argv[2] in ["cvae", "CVAE"]:
    case = "cvae"
else:
    print("Wrong model structure name!")
    exit()

BATCH = 128

if __name__ == "__main__":
    
    print("\nLoading data...", flush=True)
    test_x = np.load(sys.argv[1], allow_pickle=True)
    test_x = preprocess(test_x)
    test_dataset = ImageDataset(test_x)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH)
    
    print("\nLoading model...", flush=True)
    if case == "cvae":
        model = CVAE().cuda()
    model.load_state_dict(torch.load(sys.argv[3]))
    
    print("\nComputing losses...", flush=True)
    pred_x = []
    pred_mu = []
    pred_logvar = []
    model.eval()
    for data in test_dataloader:
        x = data.cuda()
        (mu, logvar, _), reconst_x = model(x)
        pred_x.append(reconst_x.detach().cpu().numpy())
        pred_mu.append(mu.detach().cpu().numpy())
        pred_logvar.append(logvar.detach().cpu().numpy())
    pred_x = np.concatenate(pred_x, axis=0)
    pred_mu = np.concatenate(pred_mu, axis=0)
    pred_logvar = np.concatenate(pred_logvar, axis=0)
    anomaly_mse = np.mean(np.square(test_x - pred_x).reshape(test_x.shape[0], -1), axis=1)
    anomaly_kld = -0.5 * np.mean(1 + pred_logvar - pred_mu ** 2 - np.exp(pred_logvar), axis=1)
    anomaly = anomaly_mse + anomaly_kld
    print(np.mean(anomaly))

    print("\nWriting results...", flush=True)
    with open(sys.argv[4], 'w') as f:
        f.write("id,anomaly\n")
        for i in range(anomaly.shape[0]):
            f.write('{},{}\n'.format(i + 1, anomaly[i]))
    
    if len(sys.argv) > 5:
        test_y = np.load(sys.argv[5], allow_pickle=True)
        print("ROCAUC = {:.5f}\n".format(roc_auc_score(test_y, anomaly, average="micro")), flush=True)

    if len(sys.argv) > 6:
        print("Visualizing...", flush=True)
        ranks = np.argsort(anomaly_mse)
        target = np.concatenate((ranks[:2], ranks[-2:]))
        print(anomaly_mse[target], anomaly_kld[target])
        visualize_image(test_x[target], os.path.join(sys.argv[6], "ori_{}.jpg"))
        visualize_image(pred_x[target], os.path.join(sys.argv[6], "rec_{}.jpg"))  
