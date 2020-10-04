import numpy as np
import random
import torch
import matplotlib.pyplot as plt
from PIL import Image

seed = 1114
np.random.seed(seed)
random.seed(seed)

def preprocess(image_list):
    image_list = np.array(image_list)
    image_list = np.transpose(image_list, (0, 3, 1, 2))
    image_list = image_list.astype(np.float32)
    return image_list

def visualize_image(image_list, fn):
    for i in range(len(image_list)):
        im = Image.fromarray((np.clip((np.transpose(image_list[i], (1, 2, 0)) + 1) / 2, 0, 1) * 255.).astype(np.uint8))
        im.save(fn.format(i))

def plot_scatter(feat, label, fn):
    plt.clf()
    x = feat[:, 0]
    y = feat[:, 1]
    plt.scatter(x, y, c=label)
    plt.savefig(fn)
    return
