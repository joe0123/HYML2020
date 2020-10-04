import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from utils import *
from model import AE
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

BATCH = 64
LR = 2e-5
EPOCH = 50


if __name__ == "__main__":
    train_x = np.load(sys.argv[1])
    train_x = preprocess(train_x)
    train_dataset = ImageDataset(train_x)

    model = AE().cuda()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True)
    
    for epoch in range(EPOCH):
        model.train()
        for data in train_dataloader:
            x = data.cuda()

            latents, reconst_x = model(x)
            loss = criterion(reconst_x, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print("Epoch {}/{}: Train | loss = {:.5f}".format(epoch + 1, EPOCH, loss.data), flush=True)

    torch.save(model.state_dict(), sys.argv[2])
