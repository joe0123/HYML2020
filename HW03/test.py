import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import sys
import numpy as np
np.random.seed(1114)
import torch
import torch.nn as nn
torch.cuda.empty_cache()
torch.manual_seed(1114)
from torch.utils.data import DataLoader
import pandas as pd

from data_preprocessing import test_data
from model import CNN

BATCH = 128



if __name__ == "__main__":
    print("\nLoading data...", flush=True)
    test_set = test_data(sys.argv[1])
    test_loader = DataLoader(test_set, batch_size=BATCH, shuffle=False)

    print("\nLoading model...", flush=True)
    model = CNN().cuda()
    model.load_state_dict(torch.load(sys.argv[2]))
    
    print("\nStart predicting...", flush=True)
    model.eval()
    with torch.no_grad():
        test_pred = []
        for i, data in enumerate(test_loader):
            test_pred += np.argmax(model(data.cuda()).cpu().data.numpy(), axis=1).tolist()
    print(np.unique(test_pred, return_counts=True)[1])
    
    print("\nWriting results...", flush=True)
    df = pd.DataFrame(columns=["Id", "Category"])
    df["Id"] = range(test_set.__len__())
    df["Category"] = test_pred
    df.to_csv(sys.argv[3], index=False)
