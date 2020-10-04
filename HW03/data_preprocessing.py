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



def read_imgs(dir_path, label):
    img_files = sorted(os.listdir(dir_path))

    x = []
    y = []
    for f in img_files:
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


def train_data(dir_path):
    x, y = read_imgs(dir_path, True)
    
    return ImgDataset(x, y, train_transform)


def val_data(dir_path):
    x, y = read_imgs(dir_path, True)
 
    return ImgDataset(x, y, test_transform)


def train_val_data(dir_path1, dir_path2):
    x1, y1 = read_imgs(dir_path1, True)
    x2, y2 = read_imgs(dir_path2, True)
    x = np.concatenate((x1, x2), axis=0)
    y = np.concatenate((y1, y2), axis=0)

    return ImgDataset(x, y, train_transform)

def test_data(dir_path):
    x = read_imgs(dir_path, False)
        
    return ImgDataset(x, None, test_transform)



if __name__ == "__main__":
    train_data("data/training")
    #val_data("data/validation")
