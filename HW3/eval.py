import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from data_preprocessing import val_data
from model import CNN

BATCH = 128



if __name__ == "__main__":
    print("\nLoading data...", flush=True)
    val_set = val_data(sys.argv[1])
    val_loader = DataLoader(val_set, batch_size=BATCH, shuffle=False)

    print("\nLoading model...", flush=True)
    model = CNN().cuda()
    model.load_state_dict(torch.load(sys.argv[2]))
    
    print("\nStart predicting...", flush=True)
    model.eval()
    loss = nn.CrossEntropyLoss()    # softmax + cross entropy
    y = []
    val_pred = []
    val_loss = 0
    val_acc = 0
    with torch.no_grad():
        model.eval()
        for i, data in enumerate(val_loader):
            val_pred += np.argmax(model(data[0].cuda()).cpu().data.numpy(), axis=1).tolist()
            y += data[1].cpu().data.numpy().tolist() 
    val_acc = np.mean(np.array(val_pred) == np.array(y))
    print("\nval_acc = %3.6f" % val_acc, flush=True)
    print(np.unique(val_pred, return_counts=True)[1])
    
    print("\nMaking confusion matrix...", flush=True)

    cm = np.around(confusion_matrix(y, val_pred, normalize="pred"), decimals=2)
    ax = sns.heatmap(cm, annot=True)
    ax.set(xlabel="prediction", ylabel="true")
    plt.savefig("cm.jpg")
