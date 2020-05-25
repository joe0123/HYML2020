seed = 1114
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import sys
import torch
import numpy as np
np.random.seed(seed)
import random
random.seed(seed)
import torch
torch.cuda.empty_cache()
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
from torch import nn
import torch.optim as optim

import warnings
warnings.filterwarnings('ignore')

from utils import train_data
from preprocess import Preprocess
from dataset import TwitterDataset
from model import DNN

BATCH = 256
LR = 1e-3
EPOCH = 5

def train_model(train_data, model, device, lr):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(EPOCH):
        print("\nEpoch {}/{}".format(epoch + 1, EPOCH))
        model.train()
        train_loss, train_acc = 0, 0
        for i, (x, y) in enumerate(train_data):
            x = x.to(device, dtype=torch.float)
            y = y.to(device, dtype=torch.float) # float because of the calculation in critirion
            optimizer.zero_grad()
            pred_y = model(x)
            pred_y = pred_y.squeeze() # squeeze(): to discard the dimensions equal to 1
            loss = criterion(pred_y, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pred_y[pred_y >= 0.5] = 1
            pred_y[pred_y < 0.5] = 0
            train_acc += torch.sum(torch.eq(pred_y, y)).item() / BATCH
        print("Train | Loss:{:.05f} Acc: {:.05f}".format(train_loss / len(train_data), train_acc / len(train_data)), flush=True)
        print("Saving model...")
        torch.save(model.state_dict(), "model_bow.pkl")
        print("\n-----------------------------------------------", flush=True)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\nLoading data ...", flush=True)
    train_x1, train_y1 = train_data(sys.argv[1], True)
    index = np.random.permutation(range(len(train_x1)))
    train_x1 = np.array(train_x1)[index[:50000]]
    train_y1 = np.array(train_y1)[index[:50000]]

    print("\nPreprocessing data...", flush=True)
    tmp = set()
    for st in train_x1:
        for w in st:
            tmp.add(w)
    word_index = {w: i for i, w in enumerate(tmp)}
    #import pickle
    #with open("bow.pkl", "wb") as f:
    #    pickle.dump(word_index, f)

    x = torch.zeros(len(train_x1), len(word_index))
    for i in range(len(train_x1)):
        for w in train_x1[i]:
            x[i][word_index[w]] += 1
    print(x.size())
    
    print("\nConstructing model...", flush=True)
    model = DNN(x.size(1)).to(device)
    total_param = sum(p.numel() for p in model.parameters())
    trainable_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("{} parameters with {} trainable".format(total_param, trainable_param), flush=True)
    
    print("\nStart training...", flush=True)
    train_dataset1 = TwitterDataset(x, train_y1)
    train_loader1 = torch.utils.data.DataLoader(dataset=train_dataset1, batch_size=BATCH, shuffle=True, num_workers=4) 
    train_model(train_loader1, model, device, LR)
    
    print("\nStart testing...", flush=True)
    test_x = ["today is a good day , but it is hot", "today is hot , but it is a good day"]
    test_x = [i.split() for i in test_x]
    x = torch.zeros(2, len(word_index))
    for i in range(2):
        for w in test_x[i]:
            x[i][word_index[w]] += 1
    model.load_state_dict(torch.load("model_bow.pkl"))
    model.eval()
    x = x.to(device, dtype=torch.float)
    print(model(x).cpu())
