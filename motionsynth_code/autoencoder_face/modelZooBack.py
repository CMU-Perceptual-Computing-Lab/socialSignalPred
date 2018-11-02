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
    def __init__(self, featureDim):
        super(autoencoder_first, self).__init__()
        self.encoder = nn.Sequential(
            nn.Dropout(0.25),
            nn.Conv1d(featureDim,256,25,padding=12),        #256, 73, 200
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=2, stride=2)   #256, 73, 120
        )
        self.decoder = nn.Sequential(
            #nn.MaxUnpool1d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
            nn.ConvTranspose1d(256, featureDim, 25, stride=2, padding=12, output_padding=1),
            #nn.ReLU(True)
          )  

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class autoencoder_1conv_vect(nn.Module):
    def __init__(self, featureDim):
        super(autoencoder_1conv_vect, self).__init__()
    
        self.featureDim = featureDim
        self.encode_conv_outputDim = 30

        self.encoder_conv = nn.Sequential(
            #nn.Dropout(0.25),
            nn.Conv1d(featureDim,100,3,padding=1),
            nn.ReLU(True),
            nn.BatchNorm1d(100),
            nn.MaxPool1d(kernel_size=2, stride=2),  #(batch, 100, 30)  

            # nn.Conv1d(256,256,25,padding=12),
            # nn.ReLU(True),
            # nn.BatchNorm1d(256),
            # nn.MaxPool1d(kernel_size=2, stride=2),  #(batch, 256, xx) 

            # nn.Conv1d(256,256,25,padding=12),
            # nn.ReLU(True),
            # nn.BatchNorm1d(256),
            # nn.MaxPool1d(kernel_size=2, stride=2)   #(batch, 256, 7) 
        )
        self.encoder_lin = nn.Sequential(
            nn.Linear(100*self.encode_conv_outputDim,2),
            )
        self.decoder_lin = nn.Sequential(
                nn.Linear(2,100*self.encode_conv_outputDim),
                nn.ReLU(True),
                nn.BatchNorm1d(100*self.encode_conv_outputDim),
                )
        self.decoder_conv = nn.Sequential(
            #nn.MaxUnpool1d(kernel_size=2, stride=2),
            #nn.Dropout(0.25),
            nn.ConvTranspose1d(100, featureDim, 3, stride=2, padding=1, output_padding=1),
            # nn.ReLU(True),
            # nn.BatchNorm1d(256),
            # nn.ConvTranspose1d(256, 256, 25, stride=2, padding=12, output_padding=1),
            # nn.ReLU(True),
            # nn.BatchNorm1d(256),
            # nn.ConvTranspose1d(256, featureDim, 25, stride=2, padding=12, output_padding=1),
            #nn.ReLU(True)
          )

    def forward(self, input_):
        x = self.encoder_conv(input_)
        x = x.view(-1,100*self.encode_conv_outputDim)
        x = self.encoder_lin(x)  #[batch, 2]
        
        x = self.decoder_lin(x)
        x = x.view([x.size(0), 100, self.encode_conv_outputDim])
        x = self.decoder_conv(x)
        return x


class autoencoder_1conv_vect_vae(nn.Module):
    def __init__(self,featureDim):
        super(autoencoder_1conv_vect_vae, self).__init__()

        self.featureDim = featureDim

        self.encodeDim = 20

        self.latentDim = 10

        self.encode_conv_outputDim = 60#30

        self.encoder_conv = nn.Sequential(
            #nn.Dropout(0.25),
            nn.Conv1d(featureDim,self.encodeDim,3,padding=1),
            nn.ReLU(True),
            nn.BatchNorm1d(self.encodeDim),
            nn.MaxPool1d(kernel_size=2, stride=2),  #(batch, 100, 30)  
        )
        # self.encoder_lin1 =  nn.Sequential(
        #     nn.Linear(256*30,1024),
        #     nn.ReLU(True),
        #     nn.BatchNorm1d(1024),
        #     nn.Linear(1024,512),
        #     nn.ReLU(True),
        #     nn.BatchNorm1d(512)
        # )
        self.encoder_lin21 = nn.Linear(self.encodeDim*self.encode_conv_outputDim,self.latentDim)
        self.encoder_lin22 = nn.Linear(self.encodeDim*self.encode_conv_outputDim,self.latentDim)
       
        self.decoder_lin = nn.Sequential(
                nn.Linear(self.latentDim,self.encodeDim*self.encode_conv_outputDim),
                nn.ReLU(True),
                nn.BatchNorm1d(self.encodeDim*self.encode_conv_outputDim),
                )
        
        self.decoder_conv = nn.Sequential(
            #nn.MaxUnpool1d(kernel_size=2, stride=2),
            #nn.Dropout(0.25),
            nn.ConvTranspose1d(self.encodeDim, featureDim, 3, stride=2, padding=1, output_padding=1),
            # nn.ReLU(True),
            # nn.BatchNorm1d(256),
            # nn.ConvTranspose1d(256, 256, 25, stride=2, padding=12, output_padding=1),
            # nn.ReLU(True),
            # nn.BatchNorm1d(256),
            # nn.ConvTranspose1d(256, featureDim, 25, stride=2, padding=12, output_padding=1),
            #nn.ReLU(True)
          )

    def latent_size(self):
        return 64
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            #eps = Variable(torch.randn(std.size()))#, dtype=std.dtype, layout=std.layout, device=std.device)
            #eps = Variable(std.data.new(std.size()).normal_())
            # return eps.mul(std).add_(mu)
            return eps.mul(std).add(mu)
            #return mu
            #return std.add_(mu)
        else:
            return mu

    def decode(self,z):
        
        x = self.decoder_lin(z)
        x = x.view([x.size(0), self.encodeDim, self.encode_conv_outputDim])
        x = self.decoder_conv(x)
        return x

    def forward(self, input_):  #input_: (batch, featureDim:5, frames)
        x = self.encoder_conv(input_)
        x = x.view(-1,self.encodeDim*self.encode_conv_outputDim)
        #x = self.encoder_lin1(x)

        mu = self.encoder_lin21(x)
        logvar = self.encoder_lin22(x)

        z = self.reparameterize(mu, logvar)
        #z = logvar
        
        x = self.decoder_lin(z)
        x = x.view([x.size(0), self.encodeDim, self.encode_conv_outputDim])
        x = self.decoder_conv(x)
        return x, mu, logvar



class autoencoder_3conv_vect_vae(nn.Module):
    def __init__(self,featureDim, latentDim):
        super(autoencoder_3conv_vect_vae, self).__init__()

        self.inputFeat_dim = 5
        self.input_frameLeng = 120
        self.encode_frameLeng = self.input_frameLeng /8 #15


        self.encodeDim_conv1 = 10
        self.encodeDim_conv2 = 20
        self.encodeDim_conv3 = 40
        self.encodeDim_lin1 = 100


        self.latentDim = latentDim#100

        self.encoder_conv_1 = nn.Sequential(
            #nn.Dropout(0.25),
            nn.Conv1d(self.inputFeat_dim, self.encodeDim_conv1, kernel_size=5, stride=2, padding=2),
            nn.ReLU(True),
            nn.BatchNorm1d(self.encodeDim_conv1)
        )

        self.encoder_conv_2 = nn.Sequential(
            #nn.Dropout(0.25),
            nn.Conv1d(self.encodeDim_conv1, self.encodeDim_conv2, kernel_size=5, stride=2, padding=2),
            nn.ReLU(True),
            nn.BatchNorm1d(self.encodeDim_conv2)
        )

        self.encoder_conv_3 = nn.Sequential(
            #nn.Dropout(0.25),
            nn.Conv1d(self.encodeDim_conv2, self.encodeDim_conv3, kernel_size=5, stride=2, padding=2),
            nn.ReLU(True),
            nn.BatchNorm1d(self.encodeDim_conv3)
        )

        self.encoder_lin1 =  nn.Sequential(
            nn.Linear(self.encodeDim_conv3 * self.encode_frameLeng, self.encodeDim_lin1),       #40x15 (600) -> 100
            nn.ReLU(True),
            nn.BatchNorm1d(self.encodeDim_lin1)
        )

        self.encoder_lin21 = nn.Linear(self.encodeDim_lin1,self.latentDim)
        self.encoder_lin22 = nn.Linear(self.encodeDim_lin1,self.latentDim)
       

        self.decoder_lin1 = nn.Sequential(
                nn.Linear(self.latentDim, self.encodeDim_lin1),
                nn.ReLU(True),
                nn.BatchNorm1d(self.encodeDim_lin1)
                )

        self.decoder_lin2 = nn.Sequential(
                nn.Linear(self.encodeDim_lin1, self.encodeDim_conv3 * self.encode_frameLeng),
                nn.ReLU(True),
                nn.BatchNorm1d(self.encodeDim_conv3 * self.encode_frameLeng)
                )
        
        self.decoder_conv_1 = nn.Sequential(
            #nn.MaxUnpool1d(kernel_size=2, stride=2),
            #nn.Dropout(0.25),
            nn.ConvTranspose1d(self.encodeDim_conv3, self.encodeDim_conv2, kernel_size=5, stride=2, padding=2, output_padding=1),
          )

        self.decoder_conv_2 = nn.Sequential(
            #nn.MaxUnpool1d(kernel_size=2, stride=2),
            #nn.Dropout(0.25),
            nn.ConvTranspose1d(self.encodeDim_conv2, self.encodeDim_conv1, kernel_size=5, stride=2, padding=2, output_padding=1),
          )

        self.decoder_conv_3 = nn.Sequential(
            #nn.MaxUnpool1d(kernel_size=2, stride=2),
            #nn.Dropout(0.25),
            nn.ConvTranspose1d(self.encodeDim_conv1, self.inputFeat_dim, kernel_size=5, stride=2, padding=2, output_padding=1),
          )



    def latent_size(self):
        return self.latentDim

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            #eps = Variable(torch.randn(std.size()))#, dtype=std.dtype, layout=std.layout, device=std.device)
            #eps = Variable(std.data.new(std.size()).normal_())
            # return eps.mul(std).add_(mu)
            return eps.mul(std).add(mu)
            #return mu
            #return std.add_(mu)
        else:
            return mu

    def decode(self,z):
        
        x = self.decoder_lin1(z)        #(batch, 100)

        x = self.decoder_lin2(x)        #(batch, 600)

        x = x.view([x.size(0), self.encodeDim_conv3, self.encode_frameLeng])        #(batch, 40, 15)
        x = self.decoder_conv_1(x)          #(batch, 20, 30)

        x = self.decoder_conv_2(x)          #(batch, 10, 60)

        x = self.decoder_conv_3(x)          #(batch, 10, 60)
        
        return x

    def forward(self, input_):  #input_: (batch, featureDim:5, frames:120)
        
        x = self.encoder_conv_1(input_)     #(batch, featureDim:10, 60)

        x = self.encoder_conv_2(x)      #(batch, featureDim:20, 30)

        x = self.encoder_conv_3(x)      #(batch, featureDim:40, 15)

        x = x.view(-1, self.encodeDim_conv3 * self.encode_frameLeng)    #(batch, 600)
        #x = self.encoder_lin1(x)

        x = self.encoder_lin1(x)        #(batch, 100)
       

        mu = self.encoder_lin21(x)          #(batch, latent100)
        logvar = self.encoder_lin22(x)

        z = self.reparameterize(mu, logvar)
        #z = logvar
        
        x = self.decoder_lin1(z)        #(batch, 100)

        x = self.decoder_lin2(x)        #(batch, 600)

        x = x.view([x.size(0), self.encodeDim_conv3, self.encode_frameLeng])        #(batch, 40, 15)
        x = self.decoder_conv_1(x)          #(batch, 20, 30)

        x = self.decoder_conv_2(x)          #(batch, 10, 60)

        x = self.decoder_conv_3(x)          #(batch, 10, 60)
        
        return x, mu, logvar




class autoencoder_3conv_vect_vae_conditional(nn.Module):
    def __init__(self,featureDim, latentDim):
        super(autoencoder_3conv_vect_vae_conditional, self).__init__()

        self.inputFeat_dim = 5  + 1  #+1 for the speech label
        self.input_frameLeng = 120
        self.encode_frameLeng = self.input_frameLeng /8 #15


        self.encodeDim_conv1 = 10
        self.encodeDim_conv2 = 20
        self.encodeDim_conv3 = 40
        self.encodeDim_lin1 = 100


        self.latentDim = latentDim#100

        self.encoder_conv_1 = nn.Sequential(
            #nn.Dropout(0.25),
            nn.Conv1d(self.inputFeat_dim, self.encodeDim_conv1, kernel_size=5, stride=2, padding=2),
            nn.ReLU(True),
            nn.BatchNorm1d(self.encodeDim_conv1)
        )

        self.encoder_conv_2 = nn.Sequential(
            #nn.Dropout(0.25),
            nn.Conv1d(self.encodeDim_conv1, self.encodeDim_conv2, kernel_size=5, stride=2, padding=2),
            nn.ReLU(True),
            nn.BatchNorm1d(self.encodeDim_conv2)
        )

        self.encoder_conv_3 = nn.Sequential(
            #nn.Dropout(0.25),
            nn.Conv1d(self.encodeDim_conv2, self.encodeDim_conv3, kernel_size=5, stride=2, padding=2),
            nn.ReLU(True),
            nn.BatchNorm1d(self.encodeDim_conv3)
        )

        self.encoder_lin1 =  nn.Sequential(
            nn.Linear(self.encodeDim_conv3 * self.encode_frameLeng, self.encodeDim_lin1),       #40x15 (600) -> 100
            nn.ReLU(True),
            nn.BatchNorm1d(self.encodeDim_lin1)
        )

        self.encoder_lin21 = nn.Linear(self.encodeDim_lin1,self.latentDim)
        self.encoder_lin22 = nn.Linear(self.encodeDim_lin1,self.latentDim)
       

        self.decoder_lin1 = nn.Sequential(
                nn.Linear(self.latentDim + 1, self.encodeDim_lin1),     #+1 for the label
                nn.ReLU(True),
                nn.BatchNorm1d(self.encodeDim_lin1)
                )

        self.decoder_lin2 = nn.Sequential(
                nn.Linear(self.encodeDim_lin1, self.encodeDim_conv3 * self.encode_frameLeng),
                nn.ReLU(True),
                nn.BatchNorm1d(self.encodeDim_conv3 * self.encode_frameLeng)
                )
        
        self.decoder_conv_1 = nn.Sequential(
            #nn.MaxUnpool1d(kernel_size=2, stride=2),
            #nn.Dropout(0.25),
            nn.ConvTranspose1d(self.encodeDim_conv3, self.encodeDim_conv2, kernel_size=5, stride=2, padding=2, output_padding=1),
          )

        self.decoder_conv_2 = nn.Sequential(
            #nn.MaxUnpool1d(kernel_size=2, stride=2),
            #nn.Dropout(0.25),
            nn.ConvTranspose1d(self.encodeDim_conv2, self.encodeDim_conv1, kernel_size=5, stride=2, padding=2, output_padding=1),
          )

        self.decoder_conv_3 = nn.Sequential(
            #nn.MaxUnpool1d(kernel_size=2, stride=2),
            #nn.Dropout(0.25),
            nn.ConvTranspose1d(self.encodeDim_conv1, self.inputFeat_dim, kernel_size=5, stride=2, padding=2, output_padding=1),
          )



    def latent_size(self):
        return self.latentDim

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            #eps = Variable(torch.randn(std.size()))#, dtype=std.dtype, layout=std.layout, device=std.device)
            #eps = Variable(std.data.new(std.size()).normal_())
            # return eps.mul(std).add_(mu)
            return eps.mul(std).add(mu)
            #return mu
            #return std.add_(mu)
        else:
            return mu

    def decode(self,z, speechLabel):
        
        x = self.decoder_lin1(z)        #(batch, 100)

        x = self.decoder_lin2(x)        #(batch, 600)

        x = x.view([x.size(0), self.encodeDim_conv3, self.encode_frameLeng])        #(batch, 40, 15)
        x = self.decoder_conv_1(x)          #(batch, 20, 30)

        x = self.decoder_conv_2(x)          #(batch, 10, 60)

        x = self.decoder_conv_3(x)          #(batch, 10, 60)
        
        return x

    def forward(self, input_, speechLabel):  #input_: (batch, featureDim:5, frames:120)
        
        x = self.encoder_conv_1(input_)     #(batch, featureDim:10, 60)

        x = self.encoder_conv_2(x)      #(batch, featureDim:20, 30)

        x = self.encoder_conv_3(x)      #(batch, featureDim:40, 15)

        x = x.view(-1, self.encodeDim_conv3 * self.encode_frameLeng)    #(batch, 600)
        #x = self.encoder_lin1(x)

        x = self.encoder_lin1(x)        #(batch, 100)
       

        mu = self.encoder_lin21(x)          #(batch, latent100)
        logvar = self.encoder_lin22(x)

        z = self.reparameterize(mu, logvar)
        
        z = torch.cat( (z, speechLabel), 1)  #(batch, latentDim + 1)
        #z = logvar
        
        x = self.decoder_lin1(z)        #(batch, 100)

        x = self.decoder_lin2(x)        #(batch, 600)

        x = x.view([x.size(0), self.encodeDim_conv3, self.encode_frameLeng])        #(batch, 40, 15)
        x = self.decoder_conv_1(x)          #(batch, 20, 30)

        x = self.decoder_conv_2(x)          #(batch, 10, 60)

        x = self.decoder_conv_3(x)          #(batch, 10, 60)
        
        return x, mu, logvar
