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

class BetaVAE(BetaVAE):

    def __init__(self, z_dim=10, nb_channel=3):
        super(BetaVAE, self).__init__(z_dim=z_dim, nb_channel=nb_channel)


if __name__ == '__main__':
    net = BetaVAE()
    noise = torch.randn(2, 3, 224, 224)
    # noise1 = torch.randn(2, 20)
    # print(net._encode(noise).size())
    # print(net._decode(noise1).size())
    print(net(noise)[0].size())
    # x_recon, mu, logvar = net(noise)
    # print(x.size())