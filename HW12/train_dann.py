import os
import sys
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from utils import *
from model import *

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
seed = 1114
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

BATCH = 32
LR = 1e-3
EPOCH = 2228

if __name__ == "__main__":
    print("\nLoading data...", flush=True)
    if sys.argv[2] == '1':
        source_dataset = ImageFolder(os.path.join(sys.argv[1], "val_train_data"), transform=source_train_transform)
        source_dataloader = DataLoader(source_dataset, batch_size=BATCH, shuffle=True)
        source_val_dataset = ImageFolder(os.path.join(sys.argv[1], "val_val_data"), transform=source_test_transform)
        source_val_dataloader = DataLoader(source_val_dataset, batch_size=512)
    else:
        source_dataset = ImageFolder(os.path.join(sys.argv[1], "train_data"), transform=source_train_transform)
        source_dataloader = DataLoader(source_dataset, batch_size=BATCH, shuffle=True)
    target_dataset = ImageFolder(os.path.join(sys.argv[1], "test_data"), transform=target_train_transform)
    target_dataloader = DataLoader(target_dataset, batch_size=BATCH, shuffle=True)

    print("\nBuilding model...", flush=True)
    feature_extractor = Feature_Extractor().cuda()
    label_predictor = Label_Predictor().cuda()
    domain_classifier = Domain_Classifier().cuda()

    label_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.BCEWithLogitsLoss()

    optimizer_F = optim.Adam(feature_extractor.parameters(), lr=LR)
    optimizer_L = optim.Adam(label_predictor.parameters(), lr=LR)
    optimizer_D = optim.Adam(domain_classifier.parameters(), lr=LR)

    print("\nStart training...", flush=True)
    for epoch in range(EPOCH):
        loss_D, acc_D, loss_FL, acc_FL = 0.0, [0.0, 0.0], 0.0, 0.0
        lamb = 2 / (1 + np.exp(-10 * epoch / EPOCH)) - 1
        feature_extractor.train()
        label_predictor.train()
        domain_classifier.train()
        for i, ((source_data, source_label), (target_data, _)) in enumerate(zip(source_dataloader, target_dataloader)):
            source_data = source_data.cuda()
            source_label = source_label.cuda()
            target_data = target_data.cuda()

            mixed_data = torch.cat((source_data, target_data), dim=0)
                # For the correctness of BatchNorm(statistics of source_data and target_data are different)
            domain_label = torch.zeros((mixed_data.shape[0], 1)).cuda()
            domain_label[:source_data.shape[0]] = 1

        # FIRST: train domain classifier
            feature = feature_extractor(mixed_data)
            domain_logits = domain_classifier(feature.detach())
                # Remember to detach feature to avoid updating feature extractor
            loss = domain_criterion(domain_logits, domain_label)
            loss_D += loss.item()
            loss.backward()
            optimizer_D.step()
            acc_D[0] += torch.sum(torch.sigmoid(domain_logits[:source_data.shape[0]]) >= 0.5).item()
            acc_D[1] += torch.sum(torch.sigmoid(domain_logits[source_data.shape[0]:]) < 0.5).item()

        # SECOND: train feature extractor & label predictor
            label_logits = label_predictor(feature[:source_data.shape[0]])
            domain_logits = domain_classifier(feature)
            loss = label_criterion(label_logits, source_label) - lamb * domain_criterion(domain_logits, domain_label)
                # "minus" is trying to deceive domain_classifier
            loss_FL += loss.item()
            loss.backward()
            optimizer_F.step()
            optimizer_L.step()
            acc_FL += torch.sum(torch.argmax(label_logits, dim=1) == source_label).item()

            optimizer_D.zero_grad()
            optimizer_F.zero_grad()
            optimizer_L.zero_grad()
        
        print("\nEpoch {:3d}/{:3d} (lambda: {:6.4f})\nTrain | loss_D: {:6.4f}, acc_D: {:6.4f} {:6.4f}; loss_FL: {:6.4f}, acc_FL: {:6.4f}".format(epoch + 1, EPOCH, lamb, loss_D / len(source_dataloader), acc_D[0] / len(source_dataset), acc_D[1] / len(source_dataset), loss_FL / len(source_dataloader), acc_FL / len(source_dataset)), flush=True)
        
        if sys.argv[2] == '1':
            feature_extractor.eval()
            label_predictor.eval()
            domain_classifier.eval()
            acc_D, acc_FL = 0.0, 0.0
            for i, (source_val_data, source_val_label) in enumerate(source_val_dataloader):
                source_val_data = source_val_data.cuda()
                source_val_label = source_val_label.cuda()
                feature = feature_extractor(source_val_data)
                domain_logits = domain_classifier(feature)
                label_logits = label_predictor(feature)
                acc_D += torch.sum(torch.sigmoid(domain_logits) >= 0.5).item()
                acc_FL += torch.sum(torch.argmax(label_logits, dim=1) == source_val_label).item()
            print("Valid | acc_D: {:6.4f}; acc_FL: {:6.4f}".format(acc_D / len(source_val_dataset), acc_FL / len(source_val_dataset)), flush=True)
        else:
            print("Saving model...", flush=True)
            torch.save(feature_extractor.state_dict(), "extractor_dann.pth")
            torch.save(label_predictor.state_dict(), "predictor_dann.pth")
