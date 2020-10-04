import torch
import torch.nn as nn
seed = 1114
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

class FAE(nn.Module):
    def __init__(self):
        super(FAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(3*32*32, 1024),
            nn.PReLU(),
            nn.Linear(1024, 512),
            nn.PReLU(),
            nn.Linear(512, 256),
            nn.PReLU(),
            nn.Linear(256, 128),
            nn.PReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.PReLU(),
            nn.Linear(256, 512),
            nn.PReLU(),
            nn.Linear(512, 1024),
            nn.PReLU(),
            nn.Linear(1024, 3*32*32),
            nn.Tanh(),
        )

    def forward(self, x):
        # x: [3*32*32]
        latents = self.encoder(x)
        reconst_x  = self.decoder(latents)
        return latents, reconst_x



class CAE(nn.Module):
    def __init__(self):
        super(CAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, 3, stride=1, padding=1),   # [12, 32, 32]
            nn.PReLU(),
            nn.MaxPool2d(2),    # [12, 16, 16]
            nn.Conv2d(12, 24, 3, stride=1, padding=1), # [24, 16, 16]
            nn.PReLU(),
            nn.MaxPool2d(2),    # [24, 8, 8]
            nn.Conv2d(24, 48, 3, stride=1, padding=1),    # [48, 8, 8]
            nn.PReLU(),
            nn.MaxPool2d(2), # [48, 4, 4]
            nn.Conv2d(48, 96, 3, stride=1, padding=1),    # [96, 4, 4]
            nn.PReLU(),
            nn.MaxPool2d(2), # [96, 2, 2]
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(96, 48, 3, stride=1),  # [48, 4, 4] 
            nn.PReLU(),
            nn.ConvTranspose2d(48, 24, 5, stride=1),  # [24, 8, 8] 
            nn.PReLU(),
            nn.ConvTranspose2d(24, 12, 9, stride=1),   # [12, 16, 16]
            nn.PReLU(),
            nn.ConvTranspose2d(12, 3, 17, stride=1),    # [3, 32, 32]
            nn.Tanh()
        )

    def forward(self, x):
        # x: [3, 32, 32]
        latents = self.encoder(x)
        reconst_x  = self.decoder(latents)
        return latents, reconst_x

class FVAE(nn.Module):
    def __init__(self):
        super(FVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(3*32*32, 1024),
            nn.BatchNorm1d(1024),
            nn.PReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.PReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.PReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.PReLU(),
        )
                
        self.fc_mu = nn.Sequential(
            nn.Linear(128, 64),
            #nn.PReLU()
        )
        self.fc_logvar = nn.Sequential(
            nn.Linear(128, 64),
            #nn.PReLU()
        )
        self.fc_latents = nn.Sequential(
            nn.Linear(64, 128),
            nn.PReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.PReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.PReLU(),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.PReLU(),
            nn.Linear(1024, 3*32*32),
            nn.Tanh(),
        )

 
    def reparameterize(self, mu, logvar):
        std = logvar.div(2).exp()
        eps = torch.autograd.Variable(torch.FloatTensor(std.shape).normal_().cuda())
        return mu + eps * std
    
    def forward(self, x):
        latents = self.encoder(x)
        latents = latents.reshape(latents.shape[0], -1)
        mu, logvar = self.fc_mu(latents), self.fc_logvar(latents)
        latents = self.fc_latents(self.reparameterize(mu, logvar))
        reconst_x  = self.decoder(latents)
        return (mu, logvar, latents), reconst_x


class CVAE(nn.Module):
    def __init__(self):
        super(CVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, 3, stride=1, padding=1),   # [12, 32, 32]
            nn.PReLU(),
            nn.MaxPool2d(2),    # [12, 16, 16]
            nn.Conv2d(12, 24, 3, stride=1, padding=1), # [24, 16, 16]
            nn.PReLU(),
            nn.MaxPool2d(2),    # [24, 8, 8]
            nn.Conv2d(24, 48, 3, stride=1, padding=1),    # [48, 8, 8]
            nn.PReLU(),
            nn.MaxPool2d(2), # [48, 4, 4]
        )
                
        self.fc_mu = nn.Sequential(
            nn.Linear(48*4*4, 128),
            #nn.PReLU()
        )
        self.fc_logvar = nn.Sequential(
            nn.Linear(48*4*4, 128),
            #nn.PReLU()
        )
        self.fc_latents = nn.Sequential(
            nn.Linear(128, 48*4*4),
            nn.PReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(48, 24, 5, stride=1),  # [24, 8, 8] 
            nn.PReLU(),
            nn.ConvTranspose2d(24, 12, 9, stride=1),   # [12, 16, 16]
            nn.PReLU(),
            nn.ConvTranspose2d(12, 3, 17, stride=1),    # [3, 32, 32]
            nn.Tanh()
        )

 
    def reparameterize(self, mu, logvar):
        std = logvar.div(2).exp()
        eps = torch.autograd.Variable(torch.FloatTensor(std.shape).normal_().cuda())
        return mu + eps * std
    
   
    def forward(self, x):
        latents0 = self.encoder(x)
        latents0 = latents0.reshape(latents0.shape[0], -1)
        mu, logvar = self.fc_mu(latents0), self.fc_logvar(latents0)
        latents1 = self.fc_latents(self.reparameterize(mu, logvar))
        latents1 = latents1.reshape(latents1.shape[0], 48, 4, 4)
        reconst_x  = self.decoder(latents1)
        return (mu, logvar, latents0), reconst_x
