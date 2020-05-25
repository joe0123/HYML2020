seed = 1114
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import sys
import numpy as np
np.random.seed(seed)
import random
random.seed(seed)
import torch
torch.cuda.empty_cache()
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import Adam

from data_preprocessing import train_data, val_data, train_val_data
from model import CNN
from weight_init import weight_init

BATCH = 96
LR = 1e-3
EPOCH = 100


if __name__ == "__main__":
    print("\nLoading data...", flush=True)
    if sys.argv[3] == '1':
        train_set = train_data(sys.argv[1])
        val_set = val_data(sys.argv[2])
        train_loader = DataLoader(train_set, batch_size=BATCH, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=BATCH, shuffle=False)
    else:
        train_set = train_val_data(sys.argv[1], sys.argv[2])
        train_loader = DataLoader(train_set, batch_size=BATCH, shuffle=True)

    print("\nBuilding model...", flush=True)
    model = CNN().cuda()
    #model = CNN_shallow().cuda()
    #model = DNN().cuda()
    #model.apply(weight_init)
    total_param = sum(p.numel() for p in model.parameters())
    trainable_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("{} parameters with {} trainable".format(total_param, trainable_param), flush=True)
    
    print("\nStart training...", flush=True)
    loss = nn.CrossEntropyLoss()    # softmax + cross entropy
    #class_weight = torch.tensor([1, 1.2, 1, 1, 1, 1, 1.2, 1.2, 1, 1, 1]).cuda()
    #loss = nn.CrossEntropyLoss(weight=class_weight)
    optimizer = Adam(model.parameters(), lr=LR)

    best_acc = 0.
    for epoch in range(EPOCH):
        train_loss = 0.
        train_acc = 0.
        val_loss = 0.
        val_acc = 0.

        model.train()   # Make sure the model is in the correct state (for Dropout...)
        for data in train_loader:
            optimizer.zero_grad()   # We should set zero to optimizer
            train_pred = model(data[0].cuda())
            batch_loss = loss(train_pred, data[1].cuda())   # Be careful that pred and label must be on CPU or GPU simultaneously.
            batch_loss.backward()   # Use back propagation to calculate gradient
            optimizer.step()    #  The function can be called once the gradients are computed using e.g. backward()
            
            train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
            train_loss += batch_loss.item()
        
        train_loss = train_loss / train_set.__len__() * BATCH
        train_acc /= train_set.__len__()
        if sys.argv[3] == '1':
            model.eval()
            with torch.no_grad():   # To tell Pytorch not to trace gradient
                for data in val_loader:
                    val_pred = model(data[0].cuda())
                    batch_loss = loss(val_pred, data[1].cuda())

                    val_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
                    val_loss += batch_loss.item()
        
            val_loss = val_loss / val_set.__len__() * BATCH
            val_acc /= val_set.__len__()
            print("\nEpoch = %03d/%03d, loss = %3.6f, acc = %3.6f, val_loss = %3.6f, val_acc = %3.6f" % (epoch + 1, EPOCH, train_loss, train_acc, val_loss, val_acc), flush=True)
            if val_acc > best_acc:
                print("Saving model...")
                torch.save(model.state_dict(), "model.pkl")
                best_acc = val_acc
        else:
            print("\nEpoch = %03d/%03d, loss = %3.6f, acc = %3.6f" % (epoch + 1, EPOCH, train_loss, train_acc), flush=True)
            print("Saving model...")
            torch.save(model.state_dict(), "model.pkl")
        print("\n-----------------------------------------------", flush=True)
