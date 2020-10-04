import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from data_preprocessing import get_data
from model import CNN

BATCH = 128

if __name__ == "__main__":
    print("\nLoading data...", flush=True)
    data_set = get_data(os.path.join(sys.argv[1], "validation"), range(3430))
    data_loader = DataLoader(data_set, batch_size=BATCH, shuffle=False)

    print("\nLoading model...", flush=True)
    model = CNN().cuda()
    model.load_state_dict(torch.load(sys.argv[2]))
    
    print("\nStart predicting...", flush=True)
    model.eval()
    loss = nn.CrossEntropyLoss()    # softmax + cross entropy
    y = []
    pred_y = []
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            pred_y += np.argmax(model(data[0].cuda()).cpu().data.numpy(), axis=1).tolist()
            y += data[1].cpu().data.numpy().tolist() 
    acc = np.mean(np.array(pred_y) == np.array(y))
    print("\nacc = %3.6f" % acc, flush=True)
    
    print("\nMaking confusion matrix...", flush=True)
    cm = np.around(confusion_matrix(y, pred_y, normalize="true"), decimals=2)
    ax = sns.heatmap(cm, annot=True)
    ax.set(xlabel="prediction", ylabel="true")
    plt.savefig(os.path.join(sys.argv[3], "cm.jpg"))
