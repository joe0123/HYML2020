import torch
import torch.nn as nn

seed = 1114
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

class Feature_Extractor(nn.Module):
    def __init__(self):
        super(Feature_Extractor, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=1, padding=1),  # [64, 32, 32]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),    # [64, 16, 16]

            nn.Conv2d(64, 128, 3, stride=1, padding=1), # [128, 16, 16]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),    # [128, 8, 8]

            nn.Conv2d(128, 256, 3, stride=1, padding=1),    # [256, 8, 8]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),    # [256, 4, 4]

            nn.Conv2d(256, 256, 3, stride=1, padding=1),   # [256, 4, 4]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2), # [256, 2, 2]
            
            nn.Conv2d(256, 512, 3, stride=1, padding=1),   # [512, 2, 2]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2), # [512, 1, 1]
        )

    def forward(self, x):
        x = self.conv(x)
        return x.reshape(x.shape[0], -1)

class Label_Predictor(nn.Module):
    def __init__(self):
        super(Label_Predictor, self).__init__()

        self.predictor = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            
            nn.Linear(512, 512),
            nn.ReLU(),
            
            nn.Linear(512, 10),
        )

    def forward(self, h):
        return self.predictor(h)

class Domain_Classifier(nn.Module):
    def __init__(self):
        super(Domain_Classifier, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 1),
        )

    def forward(self, h):
        return self.classifier(h)

