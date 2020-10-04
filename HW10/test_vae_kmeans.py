import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
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

BATCH = 128
NM_THRES = 279

if sys.argv[2] in ["fvae", "FVAE"]:
    case = "fvae"
elif sys.argv[2] in ["cvae", "CVAE"]:
    case = "cvae"
else:
    print("Wrong model structure name!")
    exit()


def inference(reduced_x, normal_ids, n_clusters):
    print("Clustering...", flush=True)
    pred = KMeans(n_clusters=n_clusters, random_state=14, n_jobs=-1).fit(reduced_x[normal_ids])
    anomaly = np.amin(pred.transform(reduced_x), axis=1)
    
    return anomaly


if __name__ == "__main__":
    test_x = np.load(sys.argv[1], allow_pickle=True)
    test_x = preprocess(test_x)
    if case == "fvae":
        test_x = test_x.reshape(test_x.shape[0], -1)
    test_dataset = ImageDataset(test_x)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH)

    print("\nLoading model...", flush=True)
    if case == "fvae":
        model = FVAE().cuda()
    elif case == "cvae":
        model = CVAE().cuda()
    model.load_state_dict(torch.load(sys.argv[3]))
    
    print("\nComputing losses...", flush=True)
    pred_x = []
    latents = []
    model.eval()
    pred_x = []
    pred_mu = []
    pred_logvar = []
    latents = []
    model.eval()
    for data in test_dataloader:
        x = data.cuda()
        (mu, logvar, vec), reconst_x = model(x)
        pred_x.append(reconst_x.detach().cpu().numpy())
        pred_mu.append(mu.detach().cpu().numpy())
        pred_logvar.append(logvar.detach().cpu().numpy())
        latents.append(vec.detach().cpu().reshape(vec.shape[0], -1).numpy())
    pred_x = np.concatenate(pred_x, axis=0)
    pred_mu = np.concatenate(pred_mu, axis=0)
    pred_logvar = np.concatenate(pred_logvar, axis=0)
    latents = np.concatenate(latents, axis=0)
    anomaly_vae = np.mean(np.square(test_x - pred_x).reshape(test_x.shape[0], -1), axis=1) - 0.5 * np.mean(1 + pred_logvar - pred_mu ** 2 - np.exp(pred_logvar), axis=1)
    normal_ids = np.argsort(anomaly_vae)[:NM_THRES]
    
    for n in range(2, 3):
        print(n)
        print("\nStart inference...", flush=True)
        anomaly_kmeans = inference(latents, normal_ids, n)
        print(np.mean(anomaly_kmeans))

        print("\nWriting results...", flush=True)
        anomaly = anomaly_kmeans
        with open(sys.argv[4], 'w') as f:
            f.write("id,anomaly\n")
            for i in range(anomaly.shape[0]):
                f.write('{},{}\n'.format(i + 1, anomaly[i]))
        
        if len(sys.argv) > 5:
            test_y = np.load(sys.argv[5], allow_pickle=True)
            print("ROCAUC = {:.5f}\n".format(roc_auc_score(test_y, anomaly, average="micro")), flush=True)

        if len(sys.argv) > 6:
            print("Visualizing...", flush=True)
            ranks = np.argsort(anomaly)
            visualize_image(test_x[ranks], os.path.join(sys.argv[6], "ori_{}.jpg"))
            visualize_image(pred_x[ranks], os.path.join(sys.argv[6], "rec_{}.jpg")) 
