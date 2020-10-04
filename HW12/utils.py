import cv2
import numpy as np
import torch
import torchvision.transforms as transforms

seed = 1114
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

source_train_transform = transforms.Compose([
    transforms.Grayscale(), # For canny
    transforms.Lambda(lambda x: cv2.Canny(np.array(x), 170, 300)),  # cv2 accepts not PILimage but np array
    transforms.ToPILImage(),    # Back to PILimage
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
])

target_train_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((32, 32)),    # source is 32*32, target is 28*28
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
])

source_test_transform = transforms.Compose([
    transforms.Grayscale(), # For canny
    transforms.Lambda(lambda x: cv2.Canny(np.array(x), 170, 300)),  # cv2 accepts not PILimage but np array
    transforms.ToTensor(),
])


target_test_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((32, 32)),    # source is 32*32, target is 28*28
    transforms.ToTensor(),
])

