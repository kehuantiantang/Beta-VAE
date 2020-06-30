# coding=utf-8
import torch
import torch.nn as nn

from models.vae_base import View, VAEBase



class BetaVAE(VAEBase):
    '''Model celeba according to paper'''

    def __init__(self, z_dim=10, nb_channel=3):

        super(BetaVAE, self).__init__(z_dim, nb_channel)

        self.encoder = nn.Sequential(
            nn.Conv2d(nb_channel, 32, 4, 2, 1),          # B,  32, 32, 32
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),          # B,  32, 16, 16
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),          # B,  64,  8,  8
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),          # B,  64,  4,  4
            nn.ReLU(True),
            nn.Conv2d(64, 256, 4, 1),            # B, 256,  1,  1
            nn.ReLU(True),
            View((-1, 256*1*1)),                 # B, 256
            nn.Linear(256, z_dim*2),             # B, z_dim*2
        )
        # encoder VS decoder
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 256),               # B, 256
            View((-1, 256, 1, 1)),               # B, 256,  1,  1
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 64, 4),      # B,  64,  4,  4
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 4, 2, 1), # B,  64,  8,  8
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), # B,  32, 16, 16
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1), # B,  32, 32, 32
            nn.ReLU(True),
            nn.ConvTranspose2d(32, nb_channel, 4, 2, 1),  # B, nc, 64, 64
        )

        self.weight_init()

if __name__ == '__main__':
    net = BetaVAE()
    noise = torch.randn(2, 3, 64, 64)
    x_recon, mu, logvar = net(noise)
    print(x.size())