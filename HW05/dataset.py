seed = 1114
import torch
torch.cuda.empty_cache()
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
import random
random.seed(seed)

from torch.utils.data import Dataset

class ImgDataset(Dataset):
    def __init__(self, x, y=None, transform=None):
        self.x = x
        self.y = y
        if y is not None:
            self.y = torch.LongTensor(y)    # Label is required to be a LongTensor
        self.transform = transform
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        x = self.x[index]
        if self.transform is not None:
            x = self.transform(x)
        if self.y is not None:
            y = self.y[index]
            return x, y
        else:
            return x

