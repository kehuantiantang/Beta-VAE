# coding=utf-8
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models

from model.faces_model import BetaVAE
from model.vae_base import VAEBase, View
import torch.nn.functional as F


# class BetaVAE(VAEBase):
#     def __init__(self, z_dim=10, nb_channel=3, fc_hidden1=768):
#         super(BetaVAE, self).__init__(z_dim, nb_channel)
#
#         self.fc_hidden1, self.fc_hidden2, self.CNN_embed_dim = fc_hidden1, z_dim*2, z_dim
#
#
#         # encoding components
#         resnet = models.resnet101(pretrained=True)
#
#         modules = list(resnet.children())[:-1]      # delete the last fc layer.
#         self.resnet = nn.Sequential(*modules)
#         self.fc1 = nn.Linear(resnet.fc.in_features, self.fc_hidden1)
#         self.bn1 = nn.BatchNorm1d(self.fc_hidden1, momentum=0.01)
#         self.fc2 = nn.Linear(self.fc_hidden1, self.fc_hidden2)
#         self.bn2 = nn.BatchNorm1d(self.fc_hidden2, momentum=0.01)
#         # Latent vectors mu and sigma
#         self.fc3_mu = nn.Linear(self.fc_hidden2, self.CNN_embed_dim)      # output = CNN embedding latent variables
#         self.fc3_logvar = nn.Linear(self.fc_hidden2, self.CNN_embed_dim)  # output = CNN embedding latent variables
#
#         # Sampling vector
#         self.fc4 = nn.Linear(self.CNN_embed_dim, self.fc_hidden2)
#         self.fc_bn4 = nn.BatchNorm1d(self.fc_hidden2)
#         self.fc5 = nn.Linear(self.fc_hidden2, 64 * 4 * 4)
#         self.fc_bn5 = nn.BatchNorm1d(64 * 4 * 4)
#         self.relu = nn.ReLU(inplace=True)
#
#         # Decoder
#         self.k1, self.k2, self.k3, self.k4 = (5, 5), (3, 3), (3, 3), (3, 3)      # 2d kernal size
#         self.s1, self.s2, self.s3, self.s4 = (2, 2), (2, 2), (2, 2), (2, 2)      # 2d strides
#         self.pd1, self.pd2, self.pd3, self.pd4 = (0, 0), (0, 0), (0, 0), (0, 0)  # 2d padding
#         self.convTrans6 = nn.Sequential(
#             nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=self.k4, stride=self.s4,
#                                padding=self.pd4),
#             nn.BatchNorm2d(32, momentum=0.01),
#             nn.ReLU(inplace=True),
#         )
#         self.convTrans7 = nn.Sequential(
#             nn.ConvTranspose2d(in_channels=32, out_channels=8, kernel_size=self.k3, stride=self.s3,
#                                padding=self.pd3),
#             nn.BatchNorm2d(8, momentum=0.01),
#             nn.ReLU(inplace=True),
#         )
#
#         self.convTrans8 = nn.Sequential(
#             nn.ConvTranspose2d(in_channels=8, out_channels=3, kernel_size=self.k2, stride=self.s2,
#                                padding=self.pd2),
#             nn.BatchNorm2d(3, momentum=0.01),
#             nn.Sigmoid()    # y = (y1, y2, y3) \in [0 ,1]^3
#         )
#
#
#
#
#     def _encode(self, x):
#         x = self.resnet(x)  # ResNet
#         x = x.view(x.size(0), -1)  # flatten output of conv
#
#         # FC layers
#         x = self.bn1(self.fc1(x))
#         x = self.relu(x)
#         x = self.bn2(self.fc2(x))
#         x = self.relu(x)
#         # x = F.dropout(x, p=self.drop_p, training=self.training)
#         # mu, logvar = self.fc3_mu(x), self.fc3_logvar(x)
#         # return mu, logvar
#         return x
#
#     def encoder(self, x):
#         return self._encode(x)
#
#     def decoder(self, x):
#         return self._decode(x)
#
#     def _decode(self, z):
#         x = self.relu(self.fc_bn4(self.fc4(z)))
#         x = self.relu(self.fc_bn5(self.fc5(x))).view(-1, 64, 4, 4)
#         x = self.convTrans6(x)
#         x = self.convTrans7(x)
#         x = self.convTrans8(x)
#         x = F.interpolate(x, size=(224, 224), mode='bilinear')
#         return x

class BetaVAE(VAEBase):

    def __init__(self, z_dim=10, nb_channel=3):

        super(BetaVAE, self).__init__(z_dim, nb_channel)

        # self.encoder = nn.Sequential(
        #     nn.Conv2d(nb_channel, 32, 4, 2, 1),          # B,  32, 32, 32
        #     nn.ReLU(True),
        #     nn.Conv2d(32, 32, 4, 2, 1),          # B,  32, 16, 16
        #     nn.ReLU(True),
        #     nn.Conv2d(32, 64, 4, 2, 1),          # B,  64,  8,  8
        #     nn.ReLU(True),
        #     nn.Conv2d(64, 64, 4, 2, 1),          # B,  64,  4,  4
        #     nn.ReLU(True),
        #     nn.Conv2d(64, 256, 4, 1),            # B, 256,  1,  1
        #     nn.ReLU(True),
        #     View((-1, 256*1*1)),                 # B, 256
        #     nn.Linear(256, z_dim*2),             # B, z_dim*2
        # )


        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4),  # (b x 96 x 55 x 55)
            nn.BatchNorm2d(96),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 96 x 27 x 27)
            nn.Conv2d(96, 256, 5, padding=2),  # (b x 256 x 27 x 27)
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 256 x 13 x 13)
            nn.Conv2d(256, 384, 3, padding=1),  # (b x 384 x 13 x 13)
            nn.BatchNorm2d(384),
            nn.ReLU(True),
            nn.Conv2d(384, 384, 3, padding=1),  # (b x 384 x 13 x 13)
            nn.BatchNorm2d(384),
            nn.ReLU(True),
            nn.Conv2d(384, 256, 3, padding=1),  # (b x 256 x 13 x 13)
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # (b x 256 x 6 x 6)
            nn.Conv2d(256, 256, 6),  # (b x 256 x 1 x 1)
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
            nn.Upsample(scale_factor=4, mode='bilinear',align_corners=True)
        )

        self.weight_init()


if __name__ == '__main__':
    net = BetaVAE()
    noise = torch.randn(2, 3, 256, 256)
    # noise1 = torch.randn(2, 20)
    # print(net._encode(noise).size())
    # print(net._decode(noise1).size())
    print(net(noise)[0].size())
    # x_recon, mu, logvar = net(noise)
    # print(x.size())