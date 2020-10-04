import torch
import torch.nn as nn
seed = 1114
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1),   # [64, 32, 32]
            nn.PReLU(),
            nn.MaxPool2d(2),    # [64, 16, 16]
            nn.Conv2d(64, 128, 3, stride=1, padding=1), # [128, 16, 16]
            nn.PReLU(),
            nn.MaxPool2d(2),    # [128, 8, 8]
            nn.Conv2d(128, 256, 3, stride=1, padding=1),    # [256, 8, 8]
            nn.PReLU(),
            nn.MaxPool2d(2), # [256, 4, 4]
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 5, stride=1),  # [128, 8, 8] 
            nn.PReLU(),
            nn.ConvTranspose2d(128, 64, 9, stride=1),   # [64, 16, 16]
            nn.PReLU(),
            nn.ConvTranspose2d(64, 3, 17, stride=1),    # [3, 32, 32]
            #nn.Sigmoid()
            nn.Tanh()
        )

    def forward(self, x):
        # x: [3, 32, 32]
        latents = self.encoder(x)
        reconst_x  = self.decoder(latents)
        return latents, reconst_x

class AE_best(nn.Module):
    def __init__(self):
        super(AE_best, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1),   # [64, 32, 32]
            nn.PReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),   # [64, 32, 32]
            nn.PReLU(),
            nn.MaxPool2d(2),    # [64, 16, 16]
            nn.Conv2d(64, 128, 3, stride=1, padding=1), # [128, 16, 16]
            nn.PReLU(),
            nn.Conv2d(128, 128, 3, stride=1, padding=1), # [128, 16, 16]
            nn.PReLU(),
            nn.MaxPool2d(2),    # [128, 8, 8]
            nn.Conv2d(128, 256, 3, stride=1, padding=1),    # [256, 8, 8]
            nn.PReLU(),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),    # [256, 8, 8]
            nn.PReLU(),
            nn.MaxPool2d(2), # [256, 4, 4]
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 5, stride=1),  # [128, 8, 8] 
            nn.PReLU(),
            nn.ConvTranspose2d(128, 64, 9, stride=1),   # [64, 16, 16]
            nn.PReLU(),
            nn.ConvTranspose2d(64, 3, 17, stride=1),    # [3, 32, 32]
            #nn.Sigmoid()
            nn.Tanh()
        )

    def forward(self, x):
        # x: [3, 32, 32]
        latents = self.encoder(x)
        reconst_x  = self.decoder(latents)
        return latents, reconst_x



