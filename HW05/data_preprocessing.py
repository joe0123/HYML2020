seed = 1114
import os
import cv2
import numpy as np
np.random.seed(seed)
import torch
torch.cuda.empty_cache()
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
import torchvision.transforms as transforms
import random
random.seed(seed)

from dataset import ImgDataset

train_transform = transforms.Compose([
    transforms.ToPILImage(),    # Need uint8 in array
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(45),
    transforms.RandomResizedCrop(128, scale=(0.8, 1.0)),
    transforms.Resize(128),
    transforms.ToTensor(),   # From array to tensor and normalize
])

test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(128),
    transforms.ToTensor(),   # From array to tensor and normalize
])


def read_imgs(dir_path, ids, label):
    img_files = sorted(os.listdir(dir_path))

    x = []
    y = []
    for i in ids:
        f = img_files[i]
        img = cv2.imread(os.path.join(dir_path, f))
        img = cv2.resize(img, (256, 256))
        img = cv2.GaussianBlur(img, (5, 5), 0)
        x.append(img)
        if label:
            y.append(int(f.split('_')[0]))
    #from sklearn.utils.class_weight import compute_class_weight
    #print(compute_class_weight("balanced", np.unique(y), y))
    
    if label:
        return np.array(x).astype(np.uint8), np.array(y).astype(np.uint8)
    else:
        return np.array(x).astype(np.uint8)


def get_data(dir_path, ids):
    x, y = read_imgs(dir_path, ids, True)
    return ImgDataset(x, y, test_transform)

if __name__ == "__main__":
    #train_data("data/training")
    val_data("data/validation")
