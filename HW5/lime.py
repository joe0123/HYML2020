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
from skimage.segmentation import slic
sys.path = sys.path[1:] + [sys.path[0]]
from lime import lime_image
import matplotlib.pyplot as plt

from data_preprocessing import get_data
from model import CNN

IMG_ID = [295, 645, 1034, 1492, 1663, 2010, 2386, 2451, 2501, 2626, 2944, 457]

model = None
def predict(inputs):
    model.eval()
    inputs = torch.FloatTensor(inputs).permute(0, 3, 1, 2)   # from numpy(b, h, w, c) to pytorch(b, c, h, w)
    output = model(inputs.cuda())
    return output.detach().cpu().numpy()
def segmentation(inputs):
    return slic(inputs, n_segments=100, compactness=1, sigma=1)

if __name__ == "__main__":
    print("\nLoading data...", flush=True)
    train_set = get_data(os.path.join(sys.argv[1], "validation"), IMG_ID)
    x = []
    y = []
    for i in range(len(IMG_ID)):
        x_tmp, y_tmp = train_set.__getitem__(i)
        x.append(x_tmp)
        y.append(y_tmp)
    x = torch.stack(x).cuda()
    y = torch.stack(y)

    print("\nLoading model...", flush=True)
    model = CNN().cuda()
    model.load_state_dict(torch.load(sys.argv[2]))
    model.eval()
    pred_y = np.argmax(model(x).cpu().data.numpy(), axis=1).tolist()
    
    print("\nStart LIME...", flush=True)
    fig, axs = plt.subplots(2, int(len(IMG_ID) / 2), figsize=(15, 8))
    for i, (image, label) in enumerate(zip(x.permute(0, 2, 3, 1).cpu().numpy(), y.numpy())):
        x = image.astype(np.double) # LIME has to use numpy
        explainer = lime_image.LimeImageExplainer()
        explaination = explainer.explain_instance(image=x, classifier_fn=predict, segmentation_fn=segmentation, top_labels=11)
        lime_img, mask = explaination.get_image_and_mask(label=label.item(), positive_only=False, hide_rest=False, num_features=11, min_weight=0.05)
        if i < int(len(IMG_ID) / 2):
            axs[0][i].imshow(lime_img[:, :, [2, 1, 0]])
            axs[0][i].set(xlabel="true = {}, pred = {}".format(y[i], pred_y[i]))
        else:
            axs[1][i - int(len(IMG_ID) / 2)].imshow(lime_img[:, :, [2, 1, 0]])
            axs[1][i - int(len(IMG_ID) / 2)].set(xlabel="true = {}, pred = {}".format(y[i], pred_y[i]))
    plt.savefig(os.path.join(sys.argv[3], "lime.jpg"))
    plt.close()
