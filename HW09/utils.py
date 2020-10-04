import numpy as np
import random
import torch
from scipy.ndimage import gaussian_filter
from sklearn.preprocessing import scale 
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt

seed = 1114
np.random.seed(seed)
random.seed(seed)

def preprocess(image_list):
    #image_list = np.array([gaussian_filter(img, 0.5) for img in image_list])
    image_list = np.array(image_list)
    image_list = np.transpose(image_list, (0, 3, 1, 2))
    image_list = (image_list / 255.) * 2 - 1
    image_list = image_list.astype(np.float32)
    return image_list

def inference_basic(latents):
    print("Reducing dim by KPCA...", flush=True)
    transformer = KernelPCA(n_components=200, kernel="rbf", random_state=0, n_jobs=-1)
    x_embedded = transformer.fit_transform(latents)
    print("First Reduction Shape:", x_embedded.shape)

    print("Reducing dim by TSNE...", flush=True)
    x_embedded = TSNE(n_components=2, random_state=0, n_jobs=-1).fit_transform(x_embedded)
    print("Second Reduction Shape:", x_embedded.shape)

    print("Clustering...", flush=True)
    pred = KMeans(n_clusters=2, random_state=0).fit(x_embedded)
    pred_y = np.array([int(i) for i in pred.labels_])

    return x_embedded, pred_y

def inference_improved(latents):
    print("Reducing dim by KPCA...", flush=True)
    transformer = KernelPCA(n_components=256, kernel="rbf", random_state=0, n_jobs=-1)
    x_embedded = transformer.fit_transform(latents)
    print("First Reduction Shape:", x_embedded.shape)

    print("Reducing dim by TSNE...", flush=True)
    x_embedded = TSNE(n_components=2, perplexity=30, init="pca", random_state=0, n_jobs=-1).fit_transform(x_embedded)
    print("Second Reduction Shape:", x_embedded.shape)
    
    print("Clustering...", flush=True)
    pred = KMeans(n_clusters=64, random_state=0).fit(x_embedded)
    pred_y = np.array([int(i) for i in pred.labels_])
    pred = KMeans(n_clusters=2, random_state=0).fit(pred.cluster_centers_)
    pred_y = np.array([pred.labels_[i] for i in pred_y])
    
    return x_embedded, pred_y


def inference_best(latents):
    print("Reducing dim by KPCA...", flush=True)
    transformer = KernelPCA(n_components=1024, kernel="rbf", random_state=0, n_jobs=-1)
    x_embedded = transformer.fit_transform(latents)
    #print(transformer.lambdas_)
    print("First Reduction Shape:", x_embedded.shape)

    print("Reducing dim by PCA...", flush=True)
    transformer = PCA(n_components=32, random_state=0)
    x_embedded = transformer.fit_transform(x_embedded)
    print(np.sum(transformer.explained_variance_ratio_))
    print("Second Reduction Shape:", x_embedded.shape)

    print("Reducing dim by TSNE...", flush=True)
    x_embedded = TSNE(n_components=2, perplexity=50, init="pca", random_state=0, n_jobs=-1).fit_transform(x_embedded)
    print("Third Reduction Shape:", x_embedded.shape)

    print("Clustering...", flush=True)
    pred = KMeans(n_clusters=64, random_state=0).fit(x_embedded)
    pred_y = np.array([int(i) for i in pred.labels_])
    pred = KMeans(n_clusters=2, random_state=0).fit(pred.cluster_centers_)
    pred_y = np.array([pred.labels_[i] for i in pred_y])

    return x_embedded, pred_y

def visualize_reconst(model, x, fn):
    plt.clf()
    inp = torch.Tensor(x).cuda()
    _, recs = model(inp)
    
    x = np.transpose(np.clip((x + 1) / 2 , 0, 1) * 255., (0, 2, 3, 1)).astype(np.uint8)
    plt.figure(figsize=(10,4))
    for i, img in enumerate(x):
        plt.subplot(2, 6, i + 1, xticks=[], yticks=[])
        plt.imshow(img)

    recs = recs.cpu().detach().numpy()
    recs = np.transpose(np.clip((recs + 1) / 2, 0, 1) * 255., (0, 2, 3, 1)).astype(np.uint8)
    for i, img in enumerate(recs):
        plt.subplot(2, 6, 6 + i + 1, xticks=[], yticks=[])
        plt.imshow(img)
    plt.tight_layout()
    plt.savefig(fn)


def plot_scatter(feat, label, fn):
    plt.clf()
    x = feat[:, 0]
    y = feat[:, 1]
    plt.scatter(x, y, c=label)
    plt.savefig(fn)
    return
