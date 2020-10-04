seed = 1114
import random
random.seed(seed)
import torch
torch.cuda.empty_cache()
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()  # Use super when we want to inherit Classifier and rewrite(not override) __init__
        # Input DIM = [3, 128, 128]
        self.cnn = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),   # [64, 128, 128]
                nn.BatchNorm2d(64),
                nn.PReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0),   # [64, 64, 64]

                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),   # [128, 64, 64]
                nn.BatchNorm2d(128),
                nn.PReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0),   # [128, 32, 32]
                
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),   # [256, 32, 32]
                nn.BatchNorm2d(256),
                nn.PReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0),   # [256, 16, 16]

                nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),   # [512, 16, 16]
                nn.BatchNorm2d(512),
                nn.PReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0),   # [512, 8, 8]

                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),   # [512, 8, 8]
                nn.BatchNorm2d(512),
                nn.PReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0))   # [512, 4, 4]

        self.fc = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(512*4*4, 1024),
                nn.PReLU(),
                nn.Dropout(0.2),
                nn.Linear(1024, 512),
                nn.PReLU(),
                nn.Dropout(0.1),
                nn.Linear(512, 11)
        )

    def forward(self, x):
        feature = self.cnn(x)
        return self.fc(feature.view(feature.size()[0], -1)) # -1 = calculated by machine

