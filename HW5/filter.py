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
import matplotlib.pyplot as plt

from data_preprocessing import get_data
from model import CNN

IMG_ID = [1658, 1905, 5566, 7096]
ITER = 500
LR = 1e-1

def normalize(x):
    return (x - x.min()) / (x.max() - x.min())

layer_activations = None
def filter_explaination(x, model, device, cnn_id, filter_id):
    x = x.to(device)

    model.eval()
    def _hook(model, inputs, outputs):
        global layer_activations
        layer_activations = outputs # record outputs(activation map) of the conv layer
    hook_handle = model.cnn[cnn_id].register_forward_hook(_hook)

# Filter activations: draw the filters on different images
    model(x.cuda())
    filter_activations = layer_activations[:, filter_id, :, :].detach().cpu()
    # We just want to record the filters, so we can detach from graph with gradient discarded.

# Filter visualization: which image can activate the filter most
    x = x[: 1].to(device)
    x.requires_grad_()  # Tell pytorch our input needs gradient
    optimizer = Adam([x], lr=LR)    # d(obj) / d(x)
    for it in range(ITER):
        optimizer.zero_grad()
        model(x)
        obj = -layer_activations[:, filter_id, :, :].sum()
        obj.backward()
        optimizer.step()
    filter_visualization = x.detach().cpu().squeeze()    # To discard the dimension euqal to 1
    # We just want to record x after ITERs, so we can detach from graph with gradient discarded.
    
    hook_handle.remove()    # hook_handle must be removed, or it'll always exist and work for the conv layer.
    return filter_activations, filter_visualization


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
    
    print("\nStart explaining...", flush=True)
    for cf in [[0, 25], [0, 55], [4, 64], [4, 120], [8, 45], [8, 85]]:
        print("\n{}".format(cf[1]))
        filter_activations, filter_visualization = filter_explaination(torch.stack(x), model, device, cnn_id=cf[0], filter_id=cf[1])

        print("\nDrawing filter visulaization...", flush=True)
        filter_visualization = filter_visualization[[2, 1, 0], :, :]
        plt.imshow(normalize(filter_visualization.permute(1, 2, 0)))
        plt.savefig(os.path.join(sys.argv[3], "v_{}_{}.jpg".format(cf[0], cf[1])))
        plt.close()

        print("\nDrawing filter activation...", flush=True)
        fig, axs = plt.subplots(2, len(IMG_ID), figsize=(15, 8))
        for i, img in enumerate(x):
            img = img[[2, 1, 0], :, :]
            axs[0][i].imshow(img.permute(1, 2, 0).numpy())
        for i, img in enumerate(filter_activations):
            axs[1][i].imshow(normalize(img.numpy()))
            # img[2, 1, 0] is because "transform" turn RGB to BGR
            # permute: typorch(channel, height, weight) -> matplotlib(height, weight, channel)
            # normalize: Let the value between 0 to 1, not too light not too dark
        
        plt.savefig(os.path.join(sys.argv[3], "a_{}_{}.jpg".format(cf[0], cf[1])))
        plt.close()

