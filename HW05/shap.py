import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import sys
import numpy as np
np.random.seed(1114)
import torch
import torch.nn as nn
torch.cuda.empty_cache()
torch.manual_seed(1114)
from torch.optim import Adam
sys.path = sys.path[1:] + [sys.path[0]]
import shap
import matplotlib.pyplot as plt

from data_preprocessing import get_data
from model import CNN

TOTAL_ID = range(9866)
B_SIZE = 32
IMG_ID = [7741 , 7096, 7290, 1658]


if __name__ == "__main__":
    index = np.random.permutation(TOTAL_ID)
    print("\nLoading data...", flush=True)
    train_set = get_data(os.path.join(sys.argv[1], "training"), index[:B_SIZE].tolist() + IMG_ID)
    x = []
    y = []
    for i in range(B_SIZE + len(IMG_ID)):
        x_tmp, y_tmp = train_set.__getitem__(i)
        x.append(x_tmp)
        y.append(y_tmp)
    x = torch.stack(x).cuda()

    print("\nLoading model...", flush=True)
    model = CNN().cuda()
    model.load_state_dict(torch.load(sys.argv[2]))

    print("\nComputing...", flush=True)
    e = shap.DeepExplainer(model, x[:B_SIZE])
    shap_values = e.shap_values(x[B_SIZE:])
    shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
    shap.image_plot(shap_numpy, x[B_SIZE:][:, [2, 1, 0], :, :].permute(0, 2, 3, 1).cpu().numpy(), show=False)
    plt.savefig(os.path.join(sys.argv[3], "shap.jpg"))
    plt.close()
