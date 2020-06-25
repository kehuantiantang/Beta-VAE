# coding=utf-8
from model.celea_model import BetaVAE


class BetaVAE(BetaVAE):

    def __init__(self, z_dim=10, nb_channel=3):
        super(BetaVAE, self).__init__(z_dim=z_dim, nb_channel=nb_channel)