import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
import pandas as pd

from utils import *
from model import *
from dataset import *

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
seed = 1114
np.random.seed(seed)
random.seed(seed)

BATCH = 128



def inference(latents, n_clusters, anomaly):
    pred = KMeans(n_clusters=n_clusters, random_state=0, n_jobs=-1).fit(latents)
    pred_y = np.array([i for i in pred.labels_])
    anomaly_mean = []
    for i in range(n_clusters):
        tmp = anomaly * np.where(pred_y == i, 1, 0)
        tmp = np.sort(tmp[tmp != 0])
        anomaly_mean.append(np.mean(tmp))
    anomaly_mean = np.array(anomaly_mean)
    anomaly = 1 * anomaly + 20 * np.array([anomaly_mean[pred_y[i]] for i in range(len(anomaly))])
    print(anomaly_mean)
    
    return anomaly
  

if __name__ == "__main__":
    print("\nLoading data...", flush=True)
    test_x = np.load(sys.argv[1], allow_pickle=True)
    test_x = preprocess(test_x)
    test_x = test_x.reshape(test_x.shape[0], -1)
    
    print("\nLoading result...", flush=True)
    anomaly = pd.read_csv(sys.argv[2])["anomaly"].to_numpy()

    print("\nPredicting...", flush=True)
    anomaly = inference(test_x, 10, anomaly)
    
    print("\nWriting results...", flush=True)
    with open(sys.argv[3], 'w') as f:
        f.write("id,anomaly\n")
        for i in range(anomaly.shape[0]):
            f.write('{},{}\n'.format(i + 1, anomaly[i]))
    
   
    if len(sys.argv) > 4:
        print("\nEvaluating...", flush=True)
        test_y = np.load(sys.argv[4], allow_pickle=True)
        print("ROCAUC = {:.5f}\n".format(roc_auc_score(test_y, anomaly, average="micro")), flush=True)

