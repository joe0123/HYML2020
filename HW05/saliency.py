import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import sys
import numpy as np
np.random.seed(1114)
import torch
import torch.nn as nn
torch.cuda.empty_cache()
torch.manual_seed(1114)
import matplotlib.pyplot as plt

from data_preprocessing import get_data
from model import CNN

IMG_ID = [246, 1591, 5217, 9068]



def normalize(x):
    return (x - x.min()) / (x.max() - x.min())

def compute_maps(x, y, model, device):
    model.eval()
    x = x.to(device)
    y = y.to(device)

    x.requires_grad_()  # Tell pytorch our input needs gradient

    y_pred = model(x)
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(y_pred, y)
    loss.backward()

    s = x.grad.abs().detach().cpu()
    # We just want to record x, so we can detach from graph with gradient discarded.
    
    return s


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("\nLoading data...", flush=True)
    train_set = get_data(os.path.join(sys.argv[1], "training"), IMG_ID)
    x = []
    y = []
    for i in range(len(IMG_ID)):
        x_tmp, y_tmp = train_set.__getitem__(i)
        x.append(x_tmp)
        y.append(y_tmp)

    print("\nLoading model...", flush=True)
    model = CNN().to(device)
    model.load_state_dict(torch.load(sys.argv[2]))

    print("\nMaking map...", flush=True)
    s = compute_maps(torch.stack(x), torch.stack(y), model, device)  # torch.stack can make list to tensor

    fig, axs = plt.subplots(2, len(IMG_ID), figsize=(15, 8))
    for i, img in enumerate(x):
        img = img[[2, 1, 0], :, :]
        axs[0][i].imshow(img.permute(1, 2, 0).numpy())
    for i, img in enumerate(s):
        img = img[[2, 1, 0], :, :]
        axs[1][i].imshow(normalize(img.permute(1, 2, 0)).numpy())
        # img[2, 1, 0] is because "transform" turn RGB to BGR
        # permute: typorch(channel, height, weight) -> matplotlib(height, weight, channel)
        # normalize: Let the value between 0 to 1, not too light not too dark
    plt.savefig(os.path.join(sys.argv[3], "saliency.jpg"))
    plt.close()
