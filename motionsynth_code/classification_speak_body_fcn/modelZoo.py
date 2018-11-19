"""
Model Zoo: Speaking classfication for faceParams
"""

import os
import sys
import numpy as np
import scipy.io as io

#rng = np.random.RandomState(23456)

import torch
#import torchvision
from torch import nn
from torch.autograd import Variable
import os


class naive_lstm(nn.Module):
    def __init__(self, batch_size, hidden_dim=12, feature_dim=73):
        super(naive_lstm, self).__init__()

        self.hidden_dim = hidden_dim#12
        self.feature_dim= feature_dim#200

        #self.hidden_dim = 3
        #self.feature_dim= 5
        self.num_layers = 1
        self.batch_size = batch_size
        
        self.hidden = self.init_hidden()
        self.encode = nn.Linear(self.feature_dim,self.feature_dim)
        self.dout = nn.Dropout(0.5)
        self.lstm = nn.LSTM(self.feature_dim, self.hidden_dim, num_layers = self.num_layers, dropout=0.5, batch_first=True, bidirectional=False) #batch_first=True makes the order as (batch, frames, featureNum)
        #self.proj = nn.Linear(self.hidden_dim*2,1)
        self.proj = nn.Linear(self.hidden_dim,1)
        self.out_act = nn.Sigmoid()

    def init_hidden(self):
        # return (Variable(torch.zeros(self.num_layers*2, self.batch_size, self.hidden_dim)).cuda(),
        #         Variable(torch.zeros(self.num_layers*2, self.batch_size, self.hidden_dim)).cuda())
        return (Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_dim)).cuda(),
                 Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_dim)).cuda())
    def forward(self, input_):

        #input_ dimension: (batch, timestpes, dim). Note I used batch_first for this ordering
        #lstm_out:  (batch, timesteps, hidden_dim)
        #self.hidden (tuple with two elements):  ( (1, batch, hidden_dim),  (1, batch, hidden_dim))
        input_encoded = self.dout(self.encode(input_))
        lstm_out, self.hidden = self.lstm(
                    input_encoded, self.hidden)
        
        proj = self.proj(lstm_out) #input:(batch, inputDim, outputDim ) -> output (batch, timesteps,1)
        return self.out_act(proj)



class regressor_fcn_bn(nn.Module):
    def __init__(self):
        super(regressor_fcn_bn, self).__init__()


        self.encoder = nn.Sequential(
            #nn.Dropout(0.25),
            nn.Conv1d(73,128,45,padding=22),        #256, 73, 200
            nn.ReLU(),
            nn.BatchNorm1d(128),

            #nn.Dropout(0.25),
            nn.Conv1d(128,256,25,padding=12),        #256, 73, 200
            nn.ReLU(),
            nn.BatchNorm1d(256),

            #nn.Dropout(0.25),
            nn.Conv1d(256,512,15,padding=7),        #256, 73, 200
            nn.ReLU(),
            nn.BatchNorm1d(512),

            nn.Conv1d(512,1,1),        #1d-convolution
            nn.ReLU(),
            nn.BatchNorm1d(1),
            #nn.MaxPool1d(kerne_size=2, stride=2),   #256, 73, 120
        )

        self.decoder = nn.Sequential(
            #nn.MaxUnpool1d(kernel_size=2, stride=2),
            #nn.Dropout(0.25),
            nn.ConvTranspose1d(512, 73, 25, stride=2, padding=12, output_padding=1),
            #nn.ReLU(True)
          )  
        self.out_act = nn.Sigmoid()

    def forward(self, input_):
        latent = self.encoder(input_)       #(batch, feature:510, frames)

        output = self.out_act(latent)  #each values 0~1
        #output = self.decoder(latent)
        return output




class regressor_fcn_bn_dropout(nn.Module):
    def __init__(self):
        super(regressor_fcn_bn_dropout, self).__init__()


        self.encoder = nn.Sequential(
            nn.Dropout(0.25),
            nn.Conv1d(73,128,45,padding=22),        #256, 73, 200
            nn.ReLU(),
            nn.BatchNorm1d(128),

            nn.Dropout(0.25),
            nn.Conv1d(128,256,25,padding=12),        #256, 73, 200
            nn.ReLU(),
            nn.BatchNorm1d(256),

            nn.Dropout(0.25),
            nn.Conv1d(256,512,15,padding=7),        #256, 73, 200
            nn.ReLU(),
            nn.BatchNorm1d(512),

            nn.Conv1d(512,1,1),        #1d-convolution
            nn.ReLU(),
            nn.BatchNorm1d(1),
            #nn.MaxPool1d(kerne_size=2, stride=2),   #256, 73, 120
        )

        self.decoder = nn.Sequential(
            #nn.MaxUnpool1d(kernel_size=2, stride=2),
            #nn.Dropout(0.25),
            nn.ConvTranspose1d(512, 73, 25, stride=2, padding=12, output_padding=1),
            #nn.ReLU(True)
          )  
        self.out_act = nn.Sigmoid()

    def forward(self, input_):
        latent = self.encoder(input_)       #(batch, feature:510, frames)

        output = self.out_act(latent)  #each values 0~1
        #output = self.decoder(latent)
        return output


class regressor_fcn_bn_updated(nn.Module):
    def __init__(self):
        super(regressor_fcn_bn_updated, self).__init__()


        self.encoder = nn.Sequential(
            #nn.Dropout(0.25),
            nn.Conv1d(73,128,7,padding=3),        #256, 73, 200
            nn.ReLU(),
            nn.BatchNorm1d(128),

            nn.Dropout(0.25),
            nn.Conv1d(128,256,7,padding=3),        #256, 73, 200
            nn.ReLU(),
            nn.BatchNorm1d(256),

            nn.Dropout(0.25),
            nn.Conv1d(256,512,7,padding=3),        #256, 73, 200
            nn.ReLU(),
            nn.BatchNorm1d(512),

            nn.Conv1d(512,1,1),        #1d-convolution
            #nn.ReLU(),
            #nn.BatchNorm1d(1)

            #nn.MaxPool1d(kerne_size=2, stride=2),   #256, 73, 120
        )

        self.decoder = nn.Sequential(
            #nn.MaxUnpool1d(kernel_size=2, stride=2),
            #nn.Dropout(0.25),
            nn.ConvTranspose1d(512, 73, 25, stride=2, padding=12, output_padding=1),
            #nn.ReLU(True)
          )  
        self.out_act = nn.Sigmoid()

    def forward(self, input_):
        latent = self.encoder(input_)       #(batch, feature:510, frames)

        output = self.out_act(latent)  #each values 0~1
        #output = self.decoder(latent)
        return output



class regressor_fcn_bn_updated2(nn.Module):
    def __init__(self):
        super(regressor_fcn_bn_updated2, self).__init__()


        self.encoder = nn.Sequential(
            nn.Dropout(0.25),
            nn.Conv1d(73,128,45,padding=22),        #256, 73, 200
            nn.ReLU(),
            nn.BatchNorm1d(128),

            nn.Dropout(0.25),
            nn.Conv1d(128,256,25,padding=12),        #256, 73, 200
            nn.ReLU(),
            nn.BatchNorm1d(256),

            nn.Dropout(0.25),
            nn.Conv1d(256,512,15,padding=7),        #256, 73, 200
            nn.ReLU(),
            nn.BatchNorm1d(512),

            nn.Conv1d(512,1,1),        #1d-convolution
            #nn.ReLU(),
            #nn.BatchNorm1d(1)

            #nn.MaxPool1d(kerne_size=2, stride=2),   #256, 73, 120
        )

        self.decoder = nn.Sequential(
            #nn.MaxUnpool1d(kernel_size=2, stride=2),
            #nn.Dropout(0.25),
            nn.ConvTranspose1d(512, 73, 25, stride=2, padding=12, output_padding=1),
            #nn.ReLU(True)
          )  
        self.out_act = nn.Sigmoid()

    def forward(self, input_):
        latent = self.encoder(input_)       #(batch, feature:510, frames)

        output = self.out_act(latent)  #each values 0~1
        #output = self.decoder(latent)
        return output
