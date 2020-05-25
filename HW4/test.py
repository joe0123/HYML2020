import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import sys
import torch
import numpy as np
import random
import torch
from torch import nn
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

from utils import test_data
from preprocess import Preprocess
from dataset import TwitterDataset
from model import LSTM_Net

ST_LEN = 32

def test_model(test_data, model, device):
    result = []
    model.eval()
    with torch.no_grad():
        for i, x in enumerate(test_data):
            x = x.to(device, dtype=torch.long)
            pred_y = model(x)
            pred_y = pred_y.squeeze() # squeeze(): to discard the dimensions equal to 1
            
            pred_y[pred_y >= 0.5] = 1
            pred_y[pred_y < 0.5] = 0
            result += pred_y.int().tolist()
    return result



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\nLoading data ...", flush=True) # 把 'training_label.txt' 跟 'training_nolabel.txt' 讀進來
    test_x = test_data(sys.argv[1])

    print("\nPreprocessing data...", flush=True)
    preprocess = Preprocess(ST_LEN, "w2v_model/w2v.model")
    embedding = preprocess.make_em()
    test_x = preprocess.st_index(test_x)
 
    print("\nMaking dataset...", flush=True)
    test_dataset = TwitterDataset(test_x, None)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=2048, shuffle=False, num_workers=8)
    
    print("\nLoadinging model...", flush=True)
    model = LSTM_Net(embedding).to(device)
    model.load_state_dict(torch.load(sys.argv[2]))

    print("\nStart testing...", flush=True)
    result = test_model(test_loader, model, device)

    print("\nWriting file...", flush=True)
    print("Class distribution: {}".format(np.unique(result, return_counts=True)[1]))
    df = pd.DataFrame({"id": range(test_dataset.__len__()), "label": result})
    df.to_csv(sys.argv[3], index=False)
