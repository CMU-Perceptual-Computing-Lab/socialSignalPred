import os
import sys
import numpy as np
import scipy.io as io

rng = np.random.RandomState(23456)

import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST
import os


def make_mlp(dim_list, activation='relu', batch_norm=True, dropout=0):
    layers = []
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        layers.append(nn.Linear(dim_in, dim_out))
        if batch_norm:
            layers.append(nn.BatchNorm1d(dim_out))
        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLU())
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
    return nn.Sequential(*layers)



class autoencoder_first(nn.Module):
    def __init__(self):
        super(autoencoder_first, self).__init__()
        self.encoder = nn.Sequential(
            nn.Dropout(0.25),
            nn.Conv1d(73,256,25,padding=12),        #256, 73, 200
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=2, stride=2)   #256, 73, 120
        )
        self.decoder = nn.Sequential(
            #nn.MaxUnpool1d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
            nn.ConvTranspose1d(256, 73, 25, stride=2, padding=12, output_padding=1),
            #nn.ReLU(True)
          )  

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x




class naive_mlp(nn.Module):
    def __init__(self):
        super(naive_mlp, self).__init__()

        self.feature_dim= 3
        self.output_dim = 69
        
        mlp_dims = [self.feature_dim, 20]
        activation='relu'
        dropout=0
        batch_norm = True
        self.mlp = make_mlp(
                mlp_dims,
                activation=activation,
                batch_norm=batch_norm,
                dropout=dropout
            )

        self.proj = nn.Linear(mlp_dims[-1],self.output_dim)

    def forward(self, input_):

        output = self.mlp(input_)
        return self.proj(output)



# Trajectory to Body motion
# Based on Holden's original network
class regressor_fcn(nn.Module):
    def __init__(self):
        super(regressor_fcn, self).__init__()

        self.encoder = nn.Sequential(
            nn.Dropout(0.25),
            nn.Conv1d(146,256,45,padding=22),        #256, 73, 200
            nn.ReLU(),

            nn.Dropout(0.25),
            nn.Conv1d(256,256,25,padding=12),        #256, 73, 200
            nn.ReLU(),

            nn.Dropout(0.25),
            nn.Conv1d(256,256,15,padding=7),        #256, 73, 200
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),   #256, 73, 120
        )

        self.decoder = nn.Sequential(
            #nn.MaxUnpool1d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
            nn.ConvTranspose1d(256, 73, 25, stride=2, padding=12, output_padding=1),
            #nn.ReLU(True)
          )  

    def forward(self, input_):
        latent = self.encoder(input_)
        output = self.decoder(latent)
        return output


class regressor_fcn_bn_encoder(nn.Module):
    def __init__(self):
        super(regressor_fcn_bn_encoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Dropout(0.25),
            nn.Conv1d(146,256,45,padding=22),        #256, 73, 200
            nn.ReLU(),
            nn.BatchNorm1d(256),

            nn.Dropout(0.25),
            nn.Conv1d(256,256,25,padding=12),        #256, 73, 200
            nn.ReLU(),
            nn.BatchNorm1d(256),

            nn.Dropout(0.25),
            nn.Conv1d(256,256,15,padding=7),        #256, 73, 200
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.MaxPool1d(kernel_size=2, stride=2),   #256, 73, 120
        )

        # self.decoder = nn.Sequential(
        #     #nn.MaxUnpool1d(kernel_size=2, stride=2),
        #     nn.Dropout(0.25),
        #     nn.ConvTranspose1d(256, 73, 25, stride=2, padding=12, output_padding=1),
        #     #nn.ReLU(True)
        #   )  

    def forward(self, input_):
        latent = self.encoder(input_)
        #output = self.decoder(latent)
        #return output
        return latent


class regressor_fcn_bn(nn.Module):
    def __init__(self):
        super(regressor_fcn_bn, self).__init__()

        self.encoder = nn.Sequential(
            nn.Dropout(0.25),
            nn.Conv1d(146,256,45,padding=22),        #256, 73, 200
            nn.ReLU(),
            nn.BatchNorm1d(256),

            nn.Dropout(0.25),
            nn.Conv1d(256,256,25,padding=12),        #256, 73, 200
            nn.ReLU(),
            nn.BatchNorm1d(256),

            nn.Dropout(0.25),
            nn.Conv1d(256,256,15,padding=7),        #256, 73, 200
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.MaxPool1d(kernel_size=2, stride=2),   #256, 73, 120
        )

        self.decoder = nn.Sequential(
            #nn.MaxUnpool1d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
            nn.ConvTranspose1d(256, 73, 25, stride=2, padding=12, output_padding=1),
            #nn.ReLU(True)
          )  

    def forward(self, input_):
        latent = self.encoder(input_)
        output = self.decoder(latent)
        return output



class regressor_fcn_bn_noDrop(nn.Module):
    def __init__(self):
        super(regressor_fcn_bn_noDrop, self).__init__()

        self.encoder = nn.Sequential(
            #nn.Dropout(0.25),
            nn.Conv1d(146,256,45,padding=22),        #256, 73, 200
            nn.ReLU(),
            nn.BatchNorm1d(256),

            #nn.Dropout(0.25),
            nn.Conv1d(256,256,25,padding=12),        #256, 73, 200
            nn.ReLU(),
            nn.BatchNorm1d(256),

            #nn.Dropout(0.25),
            nn.Conv1d(256,256,15,padding=7),        #256, 73, 200
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.MaxPool1d(kernel_size=2, stride=2),   #256, 73, 120
        )

        self.decoder = nn.Sequential(
            #nn.MaxUnpool1d(kernel_size=2, stride=2),
            #nn.Dropout(0.25),
            nn.ConvTranspose1d(256, 73, 25, stride=2, padding=12, output_padding=1),
            #nn.ReLU(True)
          )  

    def forward(self, input_):
        latent = self.encoder(input_)
        output = self.decoder(latent)
        return output



class regressor_fcn_bn_2(nn.Module):
    def __init__(self):
        super(regressor_fcn_bn_2, self).__init__()

        self.encoder = nn.Sequential(
            nn.Dropout(0.25),
            nn.Conv1d(146,256,45,padding=22),        #256, 73, 200
            nn.ReLU(),
            nn.BatchNorm1d(256),

            nn.Dropout(0.25),
            nn.Conv1d(256,512,25,padding=12),        #256, 73, 200
            nn.ReLU(),
            nn.BatchNorm1d(512),

            nn.Dropout(0.25),
            nn.Conv1d(512,512,15,padding=7),        #256, 73, 200
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.MaxPool1d(kernel_size=2, stride=2),   #256, 73, 120
        )

        self.decoder = nn.Sequential(
            #nn.MaxUnpool1d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
            nn.ConvTranspose1d(512, 73, 25, stride=2, padding=12, output_padding=1),
            #nn.ReLU(True)
          )  

    def forward(self, input_):
        latent = self.encoder(input_)
        output = self.decoder(latent)
        return output



class regressor_fcn_bn_3(nn.Module):
    def __init__(self):
        super(regressor_fcn_bn_3, self).__init__()

        self.encoder = nn.Sequential(
            nn.Dropout(0.25),
            nn.Conv1d(146,256,45,padding=22),        #256, 73, 200
            nn.ReLU(),
            nn.BatchNorm1d(256),

            nn.Dropout(0.25),
            nn.Conv1d(256,512,25,padding=12),        #256, 73, 200
            nn.ReLU(),
            nn.BatchNorm1d(512),

            nn.Dropout(0.25),
            nn.Conv1d(512,1024,15,padding=7),        #256, 73, 200
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.MaxPool1d(kernel_size=2, stride=2),   #256, 73, 120
        )

        self.decoder = nn.Sequential(
            #nn.MaxUnpool1d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
            nn.ConvTranspose1d(1024, 73, 25, stride=2, padding=12, output_padding=1),
            #nn.ReLU(True)
          )  

    def forward(self, input_):
        latent = self.encoder(input_)
        output = self.decoder(latent)
        return output




class regressor_fcn_bn(nn.Module):
    def __init__(self):
        super(regressor_fcn_bn, self).__init__()

        self.encoder = nn.Sequential(
            nn.Dropout(0.25),
            nn.Conv1d(146,256,45,padding=22),        #256, 73, 200
            nn.ReLU(),
            nn.BatchNorm1d(256),

            nn.Dropout(0.25),
            nn.Conv1d(256,256,25,padding=12),        #256, 73, 200
            nn.ReLU(),
            nn.BatchNorm1d(256),

            nn.Dropout(0.25),
            nn.Conv1d(256,256,15,padding=7),        #256, 73, 200
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.MaxPool1d(kernel_size=2, stride=2),   #256, 73, 120
        )

        self.decoder = nn.Sequential(
            #nn.MaxUnpool1d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
            nn.ConvTranspose1d(256, 73, 25, stride=2, padding=12, output_padding=1),
            #nn.ReLU(True)
          )  

    def forward(self, input_):
        latent = self.encoder(input_)
        output = self.decoder(latent)
        return output


class regressor_fcn_bn_noDrop(nn.Module):
    def __init__(self):
        super(regressor_fcn_bn_noDrop, self).__init__()

        self.encoder = nn.Sequential(
            #nn.Dropout(0.25),
            nn.Conv1d(146,256,45,padding=22),        #256, 73, 200
            nn.ReLU(),
            nn.BatchNorm1d(256),

            #nn.Dropout(0.25),
            nn.Conv1d(256,256,25,padding=12),        #256, 73, 200
            nn.ReLU(),
            nn.BatchNorm1d(256),

            #nn.Dropout(0.25),
            nn.Conv1d(256,256,15,padding=7),        #256, 73, 200
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.MaxPool1d(kernel_size=2, stride=2),   #256, 73, 120
        )

        self.decoder = nn.Sequential(
            #nn.MaxUnpool1d(kernel_size=2, stride=2),
            #nn.Dropout(0.25),
            nn.ConvTranspose1d(256, 73, 25, stride=2, padding=12, output_padding=1),
            #nn.ReLU(True)
          )  

    def forward(self, input_):
        latent = self.encoder(input_)
        output = self.decoder(latent)
        return output



class regressor_fcn_bn_2(nn.Module):
    def __init__(self):
        super(regressor_fcn_bn_2, self).__init__()

        self.encoder = nn.Sequential(
            nn.Dropout(0.25),
            nn.Conv1d(146,256,45,padding=22),        #256, 73, 200
            nn.ReLU(),
            nn.BatchNorm1d(256),

            nn.Dropout(0.25),
            nn.Conv1d(256,512,25,padding=12),        #256, 73, 200
            nn.ReLU(),
            nn.BatchNorm1d(512),

            nn.Dropout(0.25),
            nn.Conv1d(512,512,15,padding=7),        #256, 73, 200
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.MaxPool1d(kernel_size=2, stride=2),   #256, 73, 120
        )

        self.decoder = nn.Sequential(
            #nn.MaxUnpool1d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
            nn.ConvTranspose1d(512, 73, 25, stride=2, padding=12, output_padding=1),
            #nn.ReLU(True)
          )  

    def forward(self, input_):
        latent = self.encoder(input_)
        output = self.decoder(latent)
        return output
