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
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')

from utils import train_data
from preprocess import Preprocess
from dataset import TwitterDataset
from model import LSTM_Net
from weight_init import weight_init

ST_LEN = 32
BATCH = 96
LR = 2e-4
#LR = 5e-4
EPOCH = 30
ES = 3
SEMI_ITER = 6
SEMI_THRES = 0.2

def train_model(train_data, val_data, model, device, lr):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    val_best, es = 0, 0
    for epoch in range(EPOCH):
        print("\nEpoch {}/{}".format(epoch + 1, EPOCH))
        model.train()
        train_loss, train_acc = 0, 0
        for i, (x, y) in enumerate(train_data):
            x = x.to(device, dtype=torch.long)
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

        model.eval()
        with torch.no_grad():
            val_loss, val_acc = 0, 0
            for i, (x, y) in enumerate(val_data):
                x = x.to(device, dtype=torch.long)
                y = y.to(device, dtype=torch.float) # float because of the calculation in critirion
                pred_y = model(x)
                pred_y = pred_y.squeeze() # squeeze(): to discard the dimensions equal to 1
                loss = criterion(pred_y, y)
                
                val_loss += loss.item()
                pred_y[pred_y >= 0.5] = 1
                pred_y[pred_y < 0.5] = 0
                val_acc += torch.sum(torch.eq(pred_y, y)).item() / 96
            print("Valid | Loss:{:.05f} Acc: {:.05f}".format(val_loss / len(val_data), val_acc / len(val_data)), flush=True)

            if val_acc >= val_best:
                val_best = val_acc
                es = 0
                print("Saving model...")
                torch.save(model.state_dict(), "model.pkl")
            elif es + 1 >= ES:
                print("\n-----------------------------------------------", flush=True)
                print("\nHighest valid accuracy is {:0.5f}".format(val_best / len(val_data)), flush=True)
                break
            else:
                es += 1
            print("\n-----------------------------------------------", flush=True)

def semi_test_model(test_data, model, device, thres):
    result = []
    model.eval()
    with torch.no_grad():
        for i, x in enumerate(test_data):
            x = x.to(device, dtype=torch.long)
            pred_y = model(x)
            pred_y = pred_y.squeeze() # squeeze(): to discard the dimensions equal to 1
            result += pred_y.tolist()
    
    result = np.array(result)
    pos_i = np.random.permutation(np.argwhere(result >= 1 - thres).flatten()).tolist()
    pos_i = sorted(pos_i[:min(80000, len(pos_i))])
    neg_i = np.random.permutation(np.argwhere(result <= thres).flatten()).tolist()
    neg_i = sorted(neg_i[:min(80000, len(neg_i))])
    unk_i = sorted(list(set(range(result.shape[0])) - (set(pos_i) | set(neg_i))))

    return result, pos_i, neg_i, unk_i


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\nLoading data ...", flush=True)
    train_x1, train_y1 = train_data(sys.argv[1], True)
    train_x0 = train_data(sys.argv[2], False)


    print("\nPreprocessing data...", flush=True)
    preprocess = Preprocess(ST_LEN, "w2v_model/w2v.model")
    embedding = preprocess.make_em()
    train_x1 = preprocess.st_index(train_x1)
    train_y1 = preprocess.labels(train_y1)
    if sys.argv[3] == '1':
        train_x0 = preprocess.st_index(train_x0)
    train_x1, val_x, train_y1, val_y = train_test_split(train_x1, train_y1, test_size=0.25, shuffle=True)
    print("{} initial training data and {} validation data".format(len(train_x1), len(val_x)), flush=True)
    
    print("\nConstructing model...", flush=True)
    model = LSTM_Net(embedding).to(device)
    total_param = sum(p.numel() for p in model.parameters())
    trainable_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("{} parameters with {} trainable".format(total_param, trainable_param), flush=True)
    
    print("\nStart training...", flush=True)
    val_dataset = TwitterDataset(val_x, val_y)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=96, shuffle=False, num_workers=8)
    for it in range(SEMI_ITER):
        print("\n\n=================Iter {}/{}======================".format(it + 1, SEMI_ITER), flush=True)
        train_dataset1 = TwitterDataset(train_x1, train_y1)
        train_loader1 = torch.utils.data.DataLoader(dataset=train_dataset1, batch_size=BATCH, shuffle=True, num_workers=8)
        
        model.apply(weight_init)
        train_model(train_loader1, val_loader, model, device, LR)
        
        print("\nPredicting the unlabelled...", flush=True)
        if it < SEMI_ITER - 1 and sys.argv[3] == '1':
            model.load_state_dict(torch.load("model.pkl"))
            train_dataset0 = TwitterDataset(train_x0, None)
            train_loader0 = torch.utils.data.DataLoader(dataset=train_dataset0, batch_size=512, shuffle=False, num_workers=8)
            result, pos_i, neg_i, unk_i = semi_test_model(train_loader0, model, device, SEMI_THRES)

            if len(pos_i) + len(neg_i) < train_x1.size(0) * 0.1:
                print("Early Stopping!", flush=True)
                break
            else:
                print("Adding {} data into training data...".format(len(pos_i) + len(neg_i)), flush=True)
                train_x1 = torch.cat((train_x1, train_x0[pos_i], train_x0[neg_i]), 0)
                train_y1 = torch.cat((train_y1, preprocess.labels([1] * len(pos_i) + [0] * len(neg_i))), 0)
                train_x0 = train_x0[unk_i]
        else:
            break
