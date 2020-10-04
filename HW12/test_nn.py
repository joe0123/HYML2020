import os
import sys
import numpy as np
import random
import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import pandas as pd

from utils import *
from model import *

os.environ["CUDA_VISIBLE_DEVICES"] = '2'
seed = 1114
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

BATCH = 512

if __name__ == "__main__":
    print("\nLoading data...", flush=True)
    test_dataset = ImageFolder(os.path.join(sys.argv[1], "test_data"), transform=target_test_transform)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH, shuffle=False)

    print("\nLoading model...", flush=True)
    feature_extractor = Feature_Extractor().cuda()
    feature_extractor.load_state_dict(torch.load(sys.argv[2]))
    label_predictor = Label_Predictor().cuda()
    label_predictor.load_state_dict(torch.load(sys.argv[3]))

    print("\nStart predicting...", flush=True)
    result = []
    feature_extractor.eval()
    label_predictor.eval()
    for i, (test_data, _) in enumerate(test_dataloader):
        test_data = test_data.cuda()
        label_logits = label_predictor(feature_extractor(test_data))
        result.append(torch.argmax(label_logits, dim=1).cpu().detach().numpy())
    result = np.concatenate(result)
    print(np.unique(result, return_counts=True)[1])

    df = pd.DataFrame({"id": np.arange(0, len(result)), "label": result})
    df.to_csv(sys.argv[4],index=False)
