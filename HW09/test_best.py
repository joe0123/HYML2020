import os
import sys
import random
import numpy as np
import torch
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

BATCH = 256

if __name__ == "__main__":
    print("Loading data...", flush=True)
    test_x = np.load(sys.argv[1])
    test_x = preprocess(test_x)
    test_dataset = ImageDataset(test_x)

    print("Loading model...", flush=True)
    model = AE().cuda()
    model.load_state_dict(torch.load(sys.argv[2]))
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH)
    
    print("Making latents...", flush=True)
    model.eval()
    latents = []
    for i, x in enumerate(test_dataloader):
        x = torch.FloatTensor(x)
        vec, img = model(x.cuda())
        latents.append(vec.reshape(img.shape[0], -1).cpu().detach())
    latents = torch.cat(latents, dim=0).numpy()
    
    print("Start inference...", flush=True)
    reduced_latents, pred_y = inference_best(latents)

    print("Writing result...", flush=True)
    with open(sys.argv[3], 'w') as f:
        f.write("id,label\n")
        for i, p in enumerate(pred_y):
            f.write("{},{}\n".format(i, p))
    
    if sys.argv[4] == '1':
        print("Visualizing...", flush=True)
        visualize_reconst(model, test_x[[1, 2, 3, 6, 7, 9]], "reconst.jpg")
        plot_scatter(reduced_latents, pred_y, "scatter.jpg")
