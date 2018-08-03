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


class autoencoder_2convLayers(nn.Module):
    def __init__(self):
        super(autoencoder_2convLayers, self).__init__()
        self.encoder_l1 = nn.Sequential(
            #nn.Dropout(0.25),
            nn.Conv1d(73,256,25,padding=12),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.encoder_l2 = nn.Sequential(
            nn.Conv1d(256,256,25,padding=12),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.decoder_l1 = nn.Sequential(
            #nn.MaxUnpool1d(kernel_size=2, stride=2),
            #nn.Dropout(0.25),
            nn.ConvTranspose1d(256,256, 25, stride=2, padding=12, output_padding=1),
            nn.ReLU(True)
        )

        self.decoder_l2 = nn.Sequential(
            nn.ConvTranspose1d(256, 73, 25, stride=2, padding=12, output_padding=1)
            #nn.ReLU(True)
          )  

    def forward(self, x):       #Input: (128, 73, 240)
        x = self.encoder_l1(x)  #(128, 256, 120)
        x = self.encoder_l2(x)  #(128, 256, 60)
        x = self.decoder_l1(x)  #(128, 256, 120)
        x = self.decoder_l2(x)  #(128, 256, 120)
        return x                #Output: (128, 73, 120)




class autoencoder_3convLayers(nn.Module):
    def __init__(self):
        super(autoencoder_2convLayers, self).__init__()
        self.encoder_l1 = nn.Sequential(
            #nn.Dropout(0.25),
            nn.Conv1d(73,256,25,padding=12),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.encoder_l2 = nn.Sequential(
            nn.Conv1d(256,256,25,padding=12),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.decoder_l1 = nn.Sequential(
            #nn.MaxUnpool1d(kernel_size=2, stride=2),
            #nn.Dropout(0.25),
            nn.ConvTranspose1d(256,256, 25, stride=2, padding=12, output_padding=1),
            nn.ReLU(True)
        )

        self.decoder_l2 = nn.Sequential(
            nn.ConvTranspose1d(256, 73, 25, stride=2, padding=12, output_padding=1)
            #nn.ReLU(True)
          )  

    def forward(self, x):       #Input: (128, 73, 240)
        x = self.encoder_l1(x)  #(128, 256, 120)
        x = self.encoder_l2(x)  #(128, 256, 60)
        x = self.decoder_l1(x)  #(128, 256, 120)
        x = self.decoder_l2(x)  #(128, 256, 120)
        return x                #Output: (128, 73, 120)




# class autoencoder_3convLayers(nn.Module):
#     def __init__(self):
#         super(autoencoder_3convLayers, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Dropout(0.25),
#             nn.Conv1d(73,256,25,padding=12),
#             nn.ReLU(True),
#             nn.MaxPool1d(kernel_size=2, stride=2),

#             nn.Conv1d(256,256,25,padding=12),
#             nn.ReLU(True),
#             nn.MaxPool1d(kernel_size=2, stride=2),

#             nn.Conv1d(256,256,25,padding=12),
#             nn.ReLU(True),
#             nn.MaxPool1d(kernel_size=2, stride=2)
#         )
#         self.decoder = nn.Sequential(
#             #nn.MaxUnpool1d(kernel_size=2, stride=2),
#             nn.Dropout(0.25),
#             nn.ConvTranspose1d(256, 256, 25, stride=2, padding=12, output_padding=1),
#             nn.ReLU(True),
#             nn.ConvTranspose1d(256, 256, 25, stride=2, padding=12, output_padding=1),
#             nn.ReLU(True),
#             nn.ConvTranspose1d(256, 73, 25, stride=2, padding=12, output_padding=1),
#             #nn.ReLU(True)
#           )  

#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.decoder(x)
#         return x



class autoencoder_3convLayers_vect(nn.Module):
    def __init__(self):
        super(autoencoder_3convLayers_vect, self).__init__()
        self.encoder_conv = nn.Sequential(
            nn.Dropout(0.25),
            nn.Conv1d(73,256,25,padding=12),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=2, stride=2),  #(batch, 256, 120)  

            nn.Conv1d(256,256,25,padding=12),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(256,256,25,padding=12),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=2, stride=2)   #(batch, 256, 30) 
        )
        self.encoder_lin = nn.Linear(256*30,1024)   
        self.decoder_lin = nn.Linear(1024,256*30)   
        self.decoder_conv = nn.Sequential(
            #nn.MaxUnpool1d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
            nn.ConvTranspose1d(256, 256, 25, stride=2, padding=12, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose1d(256, 256, 25, stride=2, padding=12, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose1d(256, 73, 25, stride=2, padding=12, output_padding=1),
            #nn.ReLU(True)
          )  

    def forward(self, x):
        x = self.encoder_conv(x)
        x = x.view(-1,256*30)
        x = self.encoder_lin(x)
        
        x = self.decoder_lin(x)
        x = x.view([x.size(0), 256, 30])
        x = self.decoder_conv(x)
        return x


'''
Origial Network Code by Holden
n=window=240: input data frame length
d=70: degree of freedom of the body model
w0=25: temporal filder width
m=256:number of hidden unites
batchsize:
Network(
        DropoutLayer(amount=dropout, rng=rng),
        Conv1DLayer(filter_shape=(256, 73, 25), input_shape=(batchsize, 73, window), rng=rng),
        BiasLayer(shape=(256, 1)),
        ActivationLayer(),
        Pool1DLayer(input_shape=(batchsize, 256, window)),
    ),
    
    Network(
        Depool1DLayer(output_shape=(batchsize, 256, window), depooler='random', rng=rng),
        DropoutLayer(amount=dropout, rng=rng),
        Conv1DLayer(filter_shape=(73, 256, 25), input_shape=(batchsize, 256, window), rng=rng),
        BiasLayer(shape=(73, 1))
    )
'''


"""Vectorize the latent space"""
#Input: (batch, 73 dim, 200 frames)
class autoencoder_vectorize(nn.Module):
    def __init__(self):
        super(autoencoder_vectorize, self).__init__()
        self.encoder_conv = nn.Sequential(
            #nn.Dropout(0.25),
            nn.Conv1d(73,256,25,padding=12),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=2, stride=2), #(batch, 256, 120)  #256*120= 30720
            
        )
        self.encoder_lin = nn.Linear(256*120,1024)   

        self.decoder_lin = nn.Linear(1024,256*120)   
        self.decoder_conv = nn.Sequential(    
            #nn.Dropout(0.25),
            nn.ConvTranspose1d(256, 73, 25, stride=2, padding=12, output_padding=1),
            #nn.ReLU()
          )  

    def forward(self, x):
        x = self.encoder_conv(x)
        x = x.view(-1,256*120)
        x = self.encoder_lin(x)
        
        x = self.decoder_lin(x)
        x = x.view([x.size(0), 256, 120])
        x = self.decoder_conv(x)
        return x