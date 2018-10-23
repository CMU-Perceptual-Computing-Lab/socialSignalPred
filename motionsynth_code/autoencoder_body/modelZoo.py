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


class autoencoder_3conv_vect_vae(nn.Module):
    def __init__(self,featureDim, latentDim):
        super(autoencoder_3conv_vect_vae, self).__init__()

        self.inputFeat_dim = featureDim #73
        self.input_frameLeng = 120
        self.encode_frameLeng = self.input_frameLeng /8 #15


        self.encodeDim_conv1 = 150
        self.encodeDim_conv2 = 300
        self.encodeDim_conv3 = 600
        self.encodeDim_lin1 = 200


        self.latentDim = latentDim #200

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

    def forward(self, input_):  #input_: (batch, featureDim:73, frames:120)
        
        x = self.encoder_conv_1(input_)     #(batch, featureDim:150, 60)

        x = self.encoder_conv_2(x)      #(batch, featureDim:300, 30)

        x = self.encoder_conv_3(x)      #(batch, featureDim:600, 15)

        x = x.view(-1, self.encodeDim_conv3 * self.encode_frameLeng)    #(batch, 600)
        #x = self.encoder_lin1(x)

        x = self.encoder_lin1(x)        #(batch, 200)
       

        mu = self.encoder_lin21(x)          #(batch, latent200)
        logvar = self.encoder_lin22(x)

        z = self.reparameterize(mu, logvar)
        #z = logvar
        
        x = self.decoder_lin1(z)        #(batch, latent200)

        x = self.decoder_lin2(x)        #(batch, 9000)

        x = x.view([x.size(0), self.encodeDim_conv3, self.encode_frameLeng])        #(batch, 600, 15)
        x = self.decoder_conv_1(x)          #(batch, 300, 30)

        x = self.decoder_conv_2(x)          #(batch, 150, 60)

        x = self.decoder_conv_3(x)          #(batch, 73, 120)
        
        return x, mu, logvar


# # Reconstruction + KL divergence losses summed over all elements and batch
def vae_loss_function(recon_x, x, mu, logvar,criterion, weight_kld=1.0):
    #BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), size_average=False)
    loss = criterion(recon_x, x)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    #KLD = weight_kld * -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return loss + weight_kld* KLD, loss, KLD
    #return KLD

    


# class autoencoder_2convLayers(nn.Module):
#     def __init__(self):
#         super(autoencoder_2convLayers, self).__init__()
#         self.encoder_l1 = nn.Sequential(
#             #nn.Dropout(0.25),
#             nn.Conv1d(73,256,25,padding=12),
#             nn.ReLU(True),
#             nn.MaxPool1d(kernel_size=2, stride=2)
#         )

#         self.encoder_l2 = nn.Sequential(
#             nn.Conv1d(256,256,25,padding=12),
#             nn.ReLU(True),
#             nn.MaxPool1d(kernel_size=2, stride=2)
#         )
#         self.decoder_l1 = nn.Sequential(
#             #nn.MaxUnpool1d(kernel_size=2, stride=2),
#             #nn.Dropout(0.25),
#             nn.ConvTranspose1d(256,256, 25, stride=2, padding=12, output_padding=1),
#             nn.ReLU(True)
#         )

#         self.decoder_l2 = nn.Sequential(
#             nn.ConvTranspose1d(256, 73, 25, stride=2, padding=12, output_padding=1)
#             #nn.ReLU(True)
#           )  

#     def forward(self, x):       #Input: (128, 73, 240)
#         x = self.encoder_l1(x)  #(128, 256, 120)
#         x = self.encoder_l2(x)  #(128, 256, 60)
#         x = self.decoder_l1(x)  #(128, 256, 120)
#         x = self.decoder_l2(x)  #(128, 256, 120)
#         return x                #Output: (128, 73, 120)



# class autoencoder_2convLayers_drop(nn.Module):
#     def __init__(self):
#         super(autoencoder_2convLayers_drop, self).__init__()
#         self.encoder_l1 = nn.Sequential(
#             nn.Dropout(0.25),
#             nn.Conv1d(73,256,25,padding=12),
#             nn.ReLU(True),
#             nn.MaxPool1d(kernel_size=2, stride=2)
#         )

#         self.encoder_l2 = nn.Sequential(
#             nn.Conv1d(256,256,25,padding=12),
#             nn.ReLU(True),
#             nn.MaxPool1d(kernel_size=2, stride=2)
#         )
#         self.decoder_l1 = nn.Sequential(
#             nn.Dropout(0.25),
#             nn.ConvTranspose1d(256,256, 25, stride=2, padding=12, output_padding=1),
#             nn.ReLU(True)
#         )

#         self.decoder_l2 = nn.Sequential(
#             nn.ConvTranspose1d(256, 73, 25, stride=2, padding=12, output_padding=1)
#             #nn.ReLU(True)
#           )  

#     def forward(self, x):       #Input: (128, 73, 240)
#         x = self.encoder_l1(x)  #(128, 256, 120)
#         x = self.encoder_l2(x)  #(128, 256, 60)
#         x = self.decoder_l1(x)  #(128, 256, 120)
#         x = self.decoder_l2(x)  #(128, 256, 120)
#         return x                #Output: (128, 73, 120)



# class autoencoder_3convLayers(nn.Module):
#     def __init__(self):
#         super(autoencoder_3convLayers, self).__init__()
#         self.encoder_l1 = nn.Sequential(
#             #nn.Dropout(0.25),
#             nn.Conv1d(73,256,25,padding=12),
#             nn.ReLU(True),
#             nn.MaxPool1d(kernel_size=2, stride=2)   #(128, 256, 120)
#         )   

#         self.encoder_l2 = nn.Sequential(
#             nn.Conv1d(256,256,25,padding=12),
#             nn.ReLU(True),
#             nn.MaxPool1d(kernel_size=2, stride=2)   #(128, 256, 60)
#         )

#         self.encoder_l3 = nn.Sequential(
#             nn.Conv1d(256,256,25,padding=12),
#             nn.ReLU(True),
#             nn.MaxPool1d(kernel_size=2, stride=2)   #(128, 256, 30)
#         )


#         self.decoder_l1 = nn.Sequential(
#             #nn.Dropout(0.25),
#             nn.ConvTranspose1d(256,256, 25, stride=2, padding=12, output_padding=1),
#             nn.ReLU(True)
#         )

#         self.decoder_l2 = nn.Sequential(
#             nn.ConvTranspose1d(256, 256, 25, stride=2, padding=12, output_padding=1),
#             nn.ReLU(True)
#           )  
        
#         self.decoder_l3 = nn.Sequential(
#             nn.ConvTranspose1d(256, 73, 25, stride=2, padding=12, output_padding=1)
#             #nn.ReLU(True)
#           )  

        

#     def forward(self, x):       #Input: (128, 73, 240)
#         x = self.encoder_l1(x)  #(128, 256, 120)
#         x = self.encoder_l2(x)  #(128, 256, 60)
#         x = self.encoder_l3(x)  #(128, 256, 60)
#         x = self.decoder_l1(x)  #(128, 256, 120)
#         x = self.decoder_l2(x)  #(128, 256, 120)
#         x = self.decoder_l3(x)  #(128, 256, 120)
#         return x                #Output: (128, 73, 120)


# class autoencoder_3convLayers_drop(nn.Module):
#     def __init__(self):
#         super(autoencoder_3convLayers_drop, self).__init__()
#         self.encoder_l1 = nn.Sequential(
#             nn.Dropout(0.25),
#             nn.Conv1d(73,256,25,padding=12),
#             nn.ReLU(True),
#             nn.MaxPool1d(kernel_size=2, stride=2)   #(batch, 256, 120)
#         )   

#         self.encoder_l2 = nn.Sequential(
#             nn.Conv1d(256,256,25,padding=12),
#             nn.ReLU(True),
#             nn.MaxPool1d(kernel_size=2, stride=2)   #(batch, 256, 60)
#         )

#         self.encoder_l3 = nn.Sequential(
#             nn.Conv1d(256,256,25,padding=12),
#             nn.ReLU(True),
#             nn.MaxPool1d(kernel_size=2, stride=2)   #(batch, 256, 30) 
#         )


#         self.decoder_l1 = nn.Sequential(
#             nn.Dropout(0.25),
#             nn.ConvTranspose1d(256,256, 25, stride=2, padding=12, output_padding=1),
#             nn.ReLU(True)
#         )

#         self.decoder_l2 = nn.Sequential(
#             nn.ConvTranspose1d(256, 256, 25, stride=2, padding=12, output_padding=1),
#             nn.ReLU(True)
#           )  
        
#         self.decoder_l3 = nn.Sequential(
#             nn.ConvTranspose1d(256, 73, 25, stride=2, padding=12, output_padding=1)
#             #nn.ReLU(True)
#           )  

        

#     def forward(self, x):       #Input: (128, 73, 240)
#         x = self.encoder_l1(x)  #(128, 256, 120)
#         x = self.encoder_l2(x)  #(128, 256, 60)
#         x = self.encoder_l3(x)  #(128, 256, 60)
#         x = self.decoder_l1(x)  #(128, 256, 120)
#         x = self.decoder_l2(x)  #(128, 256, 120)
#         x = self.decoder_l3(x)  #(128, 256, 120)
#         return x                #Output: (128, 73, 120)


# class autoencoder_3conv_vect3_64(nn.Module):
#     def __init__(self):
#         super(autoencoder_3conv_vect3_64, self).__init__()
#         self.encoder_conv = nn.Sequential(
#             #nn.Dropout(0.25),
#             nn.Conv1d(73,256,25,padding=12),
#             nn.ReLU(True),
#             nn.BatchNorm1d(256),
#             nn.MaxPool1d(kernel_size=2, stride=2),  #(batch, 256, 120)  

#             nn.Conv1d(256,256,25,padding=12),
#             nn.ReLU(True),
#             nn.BatchNorm1d(256),
#             nn.MaxPool1d(kernel_size=2, stride=2),  #(batch, 256, 60) 

#             nn.Conv1d(256,256,25,padding=12),
#             nn.ReLU(True),
#             nn.BatchNorm1d(256),
#             nn.MaxPool1d(kernel_size=2, stride=2)   #(batch, 256, 30) 
#         )
#         self.encoder_lin = nn.Sequential(
#             nn.Linear(256*30,1024),
#             nn.ReLU(True),
#             nn.BatchNorm1d(1024),
#             nn.Linear(1024,512),
#             nn.ReLU(True),
#             nn.BatchNorm1d(512),
#             nn.Linear(512,64)
#             )
#         self.decoder_lin = nn.Sequential(
#                 nn.Linear(64,512),
#                 nn.ReLU(True),
#                 nn.BatchNorm1d(512),
#                 nn.Linear(512,1024),
#                 nn.ReLU(True),
#                 nn.BatchNorm1d(1024),
#                 nn.Linear(1024,256*30),
#                 nn.ReLU(True),
#                 nn.BatchNorm1d(256*30),
#                 )
#         self.decoder_conv = nn.Sequential(
#             #nn.MaxUnpool1d(kernel_size=2, stride=2),
#             #nn.Dropout(0.25),
#             nn.ConvTranspose1d(256, 256, 25, stride=2, padding=12, output_padding=1),
#             nn.ReLU(True),
#             nn.BatchNorm1d(256),
#             nn.ConvTranspose1d(256, 256, 25, stride=2, padding=12, output_padding=1),
#             nn.ReLU(True),
#             nn.BatchNorm1d(256),
#             nn.ConvTranspose1d(256, 73, 25, stride=2, padding=12, output_padding=1),
#             #nn.ReLU(True)
#           )

#     def forward(self, x):
#         x = self.encoder_conv(x)
#         x = x.view(-1,256*30)
#         x = self.encoder_lin(x)
        
#         x = self.decoder_lin(x)
#         x = x.view([x.size(0), 256, 30])
#         x = self.decoder_conv(x)
#         return x


# class autoencoder_3conv_vect3_8(nn.Module):
#     def __init__(self):
#         super(autoencoder_3conv_vect3_8, self).__init__()
#         self.encoder_conv = nn.Sequential(
#             #nn.Dropout(0.25),
#             nn.Conv1d(73,256,25,padding=12),
#             nn.ReLU(True),
#             nn.BatchNorm1d(256),
#             nn.MaxPool1d(kernel_size=2, stride=2),  #(batch, 256, 120)  

#             nn.Conv1d(256,256,25,padding=12),
#             nn.ReLU(True),
#             nn.BatchNorm1d(256),
#             nn.MaxPool1d(kernel_size=2, stride=2),  #(batch, 256, 60) 

#             nn.Conv1d(256,256,25,padding=12),
#             nn.ReLU(True),
#             nn.BatchNorm1d(256),
#             nn.MaxPool1d(kernel_size=2, stride=2)   #(batch, 256, 30) 
#         )
#         self.encoder_lin = nn.Sequential(
#             nn.Linear(256*30,1024),
#             nn.ReLU(True),
#             nn.BatchNorm1d(1024),
#             nn.Linear(1024,128),
#             nn.ReLU(True),
#             nn.BatchNorm1d(128),
#             nn.Linear(128,8)
#             )
#         self.decoder_lin = nn.Sequential(
#                 nn.Linear(8,128),
#                 nn.ReLU(True),
#                 nn.BatchNorm1d(128),
#                 nn.Linear(128,1024),
#                 nn.ReLU(True),
#                 nn.BatchNorm1d(1024),
#                 nn.Linear(1024,256*30),
#                 nn.ReLU(True),
#                 nn.BatchNorm1d(256*30),
#                 )
#         self.decoder_conv = nn.Sequential(
#             #nn.MaxUnpool1d(kernel_size=2, stride=2),
#             #nn.Dropout(0.25),
#             nn.ConvTranspose1d(256, 256, 25, stride=2, padding=12, output_padding=1),
#             nn.ReLU(True),
#             nn.BatchNorm1d(256),
#             nn.ConvTranspose1d(256, 256, 25, stride=2, padding=12, output_padding=1),
#             nn.ReLU(True),
#             nn.BatchNorm1d(256),
#             nn.ConvTranspose1d(256, 73, 25, stride=2, padding=12, output_padding=1),
#             #nn.ReLU(True)
#           )

#     def forward(self, x):
#         x = self.encoder_conv(x)
#         x = x.view(-1,256*30)
#         x = self.encoder_lin(x)
        
#         x = self.decoder_lin(x)
#         x = x.view([x.size(0), 256, 30])
#         x = self.decoder_conv(x)
#         return x

# class autoencoder_3conv_vect3_2(nn.Module):
#     def __init__(self):
#         super(autoencoder_3conv_vect3_2, self).__init__()
#         self.encoder_conv = nn.Sequential(
#             #nn.Dropout(0.25),
#             nn.Conv1d(73,256,25,padding=12),
#             nn.ReLU(True),
#             nn.BatchNorm1d(256),
#             nn.MaxPool1d(kernel_size=2, stride=2),  #(batch, 256, 120)  

#             nn.Conv1d(256,256,25,padding=12),
#             nn.ReLU(True),
#             nn.BatchNorm1d(256),
#             nn.MaxPool1d(kernel_size=2, stride=2),  #(batch, 256, 60) 

#             nn.Conv1d(256,256,25,padding=12),
#             nn.ReLU(True),
#             nn.BatchNorm1d(256),
#             nn.MaxPool1d(kernel_size=2, stride=2)   #(batch, 256, 30) 
#         )
#         self.encoder_lin = nn.Sequential(
#             nn.Linear(256*30,1024),
#             nn.ReLU(True),
#             nn.BatchNorm1d(1024),
#             nn.Linear(1024,128),
#             nn.ReLU(True),
#             nn.BatchNorm1d(128),
#             nn.Linear(128,2)
#             )
#         self.decoder_lin = nn.Sequential(
#                 nn.Linear(2,128),
#                 nn.ReLU(True),
#                 nn.BatchNorm1d(128),
#                 nn.Linear(128,1024),
#                 nn.ReLU(True),
#                 nn.BatchNorm1d(1024),
#                 nn.Linear(1024,256*30),
#                 nn.ReLU(True),
#                 nn.BatchNorm1d(256*30),
#                 )
#         self.decoder_conv = nn.Sequential(
#             #nn.MaxUnpool1d(kernel_size=2, stride=2),
#             #nn.Dropout(0.25),
#             nn.ConvTranspose1d(256, 256, 25, stride=2, padding=12, output_padding=1),
#             nn.ReLU(True),
#             nn.BatchNorm1d(256),
#             nn.ConvTranspose1d(256, 256, 25, stride=2, padding=12, output_padding=1),
#             nn.ReLU(True),
#             nn.BatchNorm1d(256),
#             nn.ConvTranspose1d(256, 73, 25, stride=2, padding=12, output_padding=1),
#             #nn.ReLU(True)
#           )

#     def forward(self, x):
#         x = self.encoder_conv(x)
#         x = x.view(-1,256*30)
#         x = self.encoder_lin(x)
        
#         x = self.decoder_lin(x)
#         x = x.view([x.size(0), 256, 30])
#         x = self.decoder_conv(x)
#         return x


# class autoencoder_3convLayers_vect(nn.Module):
#     def __init__(self, frameLeng=240):
#         super(autoencoder_3convLayers_vect, self).__init__()

#         #self.m_frameLeng = frameLeng
#         if frameLeng==160:
#             finalFrameLeng = 20
#         else:# frameLeng==240:
#             finalFrameLeng = 30
#         #self.m_finalFrameLeng = finalFrameLeng


#         self.encoder_conv = nn.Sequential(
#             #nn.Dropout(0.25),
#             nn.Conv1d(73,256,25,padding=12),
#             nn.ReLU(True),
#             nn.BatchNorm1d(256),
#             nn.MaxPool1d(kernel_size=2, stride=2),  #(batch, 256, 120)  

#             nn.Conv1d(256,256,25,padding=12),
#             nn.ReLU(True),
#             nn.BatchNorm1d(256),
#             nn.MaxPool1d(kernel_size=2, stride=2),  #(batch, 256, 60) 

#             nn.Conv1d(256,256,25,padding=12),
#             nn.ReLU(True),
#             nn.BatchNorm1d(256),
#             nn.MaxPool1d(kernel_size=2, stride=2)   #(batch, 256, 30) 
#         )
#         self.encoder_lin = nn.Sequential(
#             nn.Linear(256*finalFrameLeng,1024)
#             )
#         self.decoder_lin = nn.Sequential(
#                 nn.Linear(1024,256*finalFrameLeng)
#                 )
#         self.decoder_conv = nn.Sequential(
#             #nn.MaxUnpool1d(kernel_size=2, stride=2),
#             #nn.Dropout(0.25),
#             nn.ConvTranspose1d(256, 256, 25, stride=2, padding=12, output_padding=1),
#             nn.ReLU(True),
#             nn.BatchNorm1d(256),
#             nn.ConvTranspose1d(256, 256, 25, stride=2, padding=12, output_padding=1),
#             nn.ReLU(True),
#             nn.BatchNorm1d(256),
#             nn.ConvTranspose1d(256, 73, 25, stride=2, padding=12, output_padding=1),
#             #nn.ReLU(True)
#           )

#     def forward(self, x):
#         x = self.encoder_conv(x)
#         #x = x.view(-1,256*30)
#         x = x.view(x.size(0),-1)
#         x = self.encoder_lin(x)
        
#         x = self.decoder_lin(x)
#         #x = x.view([x.size(0), 256, 30])
#         x = x.view([x.size(0), 256, -1])
#         x = self.decoder_conv(x)
#         return x



# class autoencoder_3conv_vae_64(nn.Module):
#     def __init__(self):
#         super(autoencoder_3conv_vae_64, self).__init__()

        
#         self.encoder_conv = nn.Sequential(
#             #nn.Dropout(0.25),
#             nn.Conv1d(73,256,25,padding=12),
#             nn.ReLU(True),
#             nn.BatchNorm1d(256),
#             nn.MaxPool1d(kernel_size=2, stride=2),  #(batch, 256, 120)  

#             nn.Conv1d(256,256,25,padding=12),
#             nn.ReLU(True),
#             nn.BatchNorm1d(256),
#             nn.MaxPool1d(kernel_size=2, stride=2),  #(batch, 256, 60) 

#             nn.Conv1d(256,256,25,padding=12),
#             nn.ReLU(True),
#             nn.BatchNorm1d(256),
#             nn.MaxPool1d(kernel_size=2, stride=2)   #(batch, 256, 30) 
#         )
#         self.encoder_lin1 =  nn.Sequential(
#             nn.Linear(256*30,1024),
#             nn.ReLU(True),
#             nn.BatchNorm1d(1024),
#             nn.Linear(1024,512),
#             nn.ReLU(True),
#             nn.BatchNorm1d(512)
#         )
#         self.encoder_lin21 = nn.Linear(512,64)
#         self.encoder_lin22 = nn.Linear(512,64)
       
#         self.decoder_lin1 = nn.Sequential(
#             nn.Linear(64,512),
#             nn.ReLU(True),
#             nn.BatchNorm1d(512),
#             nn.Linear(512,1024),
#             nn.ReLU(True),
#             nn.BatchNorm1d(1024)
#         )
#         self.decoder_lin2 = nn.Sequential(
#             nn.Linear(1024,256*30),
#             nn.ReLU(True),
#             nn.BatchNorm1d(256*30)
#         )
#         self.decoder_conv = nn.Sequential(
#             #nn.MaxUnpool1d(kernel_size=2, stride=2),
#             #nn.Dropout(0.25),
#             nn.ConvTranspose1d(256, 256, 25, stride=2, padding=12, output_padding=1),
#             nn.ReLU(True),
#             nn.BatchNorm1d(256),
#             nn.ConvTranspose1d(256, 256, 25, stride=2, padding=12, output_padding=1),
#             nn.ReLU(True),
#             nn.BatchNorm1d(256),
#             nn.ConvTranspose1d(256, 73, 25, stride=2, padding=12, output_padding=1),
#             #nn.ReLU(True)
#           )

#     def latent_size(self):
#         return 64
#     def reparameterize(self, mu, logvar):
#         if self.training:
#             std = torch.exp(0.5*logvar)
#             eps = torch.randn_like(std)
#             #eps = Variable(torch.randn(std.size()))#, dtype=std.dtype, layout=std.layout, device=std.device)
#             #eps = Variable(std.data.new(std.size()).normal_())
#             # return eps.mul(std).add_(mu)
#             return eps.mul(std).add(mu)
#             #return mu
#             #return std.add_(mu)
#         else:
#             return mu

#     def decode(self,z):
        
#         x = self.decoder_lin1(z)
#         x = self.decoder_lin2(x)
#         x = x.view([x.size(0), 256, 30])
#         x = self.decoder_conv(x)
#         return x

#     def forward(self, x):
#         x = self.encoder_conv(x)
#         x = x.view(-1,256*30)
#         x = self.encoder_lin1(x)


#         mu = self.encoder_lin21(x)
#         logvar = self.encoder_lin22(x)

#         z = self.reparameterize(mu, logvar)
#         #z = logvar
        
#         x = self.decoder_lin1(z)
#         x = self.decoder_lin2(x)
#         x = x.view([x.size(0), 256, 30])
#         x = self.decoder_conv(x)
#         return x, mu, logvar

# class autoencoder_3conv_vae(nn.Module):
#     def __init__(self, frameLeng=240):
#         super(autoencoder_3conv_vae, self).__init__()

#         assert type(frameLeng) == int

#         #self.m_frameLeng = frameLeng
#         if frameLeng==160:
#             finalFrameLeng = 20
#         else:# frameLeng==240:
#             finalFrameLeng = 30
#         #self.m_finalFrameLeng = finalFrameLeng

#         self.encoder_conv = nn.Sequential(
#             #nn.Dropout(0.25),
#             nn.Conv1d(73,256,25,padding=12),
#             nn.ReLU(True),
#             nn.BatchNorm1d(256),
#             nn.MaxPool1d(kernel_size=2, stride=2),  #(batch, 256, 120)  

#             nn.Conv1d(256,256,25,padding=12),
#             nn.ReLU(True),
#             nn.BatchNorm1d(256),
#             nn.MaxPool1d(kernel_size=2, stride=2),  #(batch, 256, 60) 

#             nn.Conv1d(256,256,25,padding=12),
#             nn.ReLU(True),
#             nn.BatchNorm1d(256),
#             nn.MaxPool1d(kernel_size=2, stride=2)   #(batch, 256, 30) or (batch, 256, 20)
#         )
#         self.encoder_lin1 =  nn.Sequential(
#             nn.Linear(256*finalFrameLeng,1024),
#             nn.ReLU(True),
#             nn.BatchNorm1d(1024)
#         )
#         self.encoder_lin21 = nn.Linear(1024,512)
#         self.encoder_lin22 = nn.Linear(1024,512)
       
#         self.decoder_lin1 = nn.Sequential(
#             nn.Linear(512,1024),
#             nn.ReLU(True),
#             nn.BatchNorm1d(1024)
#         )
#         self.decoder_lin2 = nn.Sequential(
#             nn.Linear(1024,256*finalFrameLeng),
#             nn.ReLU(True),
#             nn.BatchNorm1d(256*finalFrameLeng)
#         )
#         self.decoder_conv = nn.Sequential(
#             #nn.MaxUnpool1d(kernel_size=2, stride=2),
#             #nn.Dropout(0.25),
#             nn.ConvTranspose1d(256, 256, 25, stride=2, padding=12, output_padding=1),
#             nn.ReLU(True),
#             nn.BatchNorm1d(256),
#             nn.ConvTranspose1d(256, 256, 25, stride=2, padding=12, output_padding=1),
#             nn.ReLU(True),
#             nn.BatchNorm1d(256),
#             nn.ConvTranspose1d(256, 73, 25, stride=2, padding=12, output_padding=1),
#             #nn.ReLU(True)
#           )

#     def latent_size(self):
#         return 512

#     def reparameterize(self, mu, logvar):
#         if self.training:
#             std = torch.exp(0.5*logvar)
#             eps = torch.randn_like(std)
#             #eps = Variable(torch.randn(std.size()))#, dtype=std.dtype, layout=std.layout, device=std.device)
#             #eps = Variable(std.data.new(std.size()).normal_())
#             # return eps.mul(std).add_(mu)
#             return eps.mul(std).add(mu)
#             #return mu
#             #return std.add_(mu)
#         else:
#             return mu

#     def decode(self,z):
        
#         x = self.decoder_lin1(z)
#         x = self.decoder_lin2(x)
#         #x = x.view([x.size(0), 256, 30])
#         x = x.view([x.size(0), 256, -1])
#         x = self.decoder_conv(x)
#         return x

#     def forward(self, x):
#         x = self.encoder_conv(x)
#         #x = x.view(x.size(0),256*self.m_finalFrameLeng)
#         x = x.view(x.size(0),-1)
#         x = self.encoder_lin1(x)

#         mu = self.encoder_lin21(x)
#         logvar = self.encoder_lin22(x)

#         z = self.reparameterize(mu, logvar)
#         #z = logvar
        
#         x = self.decoder_lin1(z)
#         x = self.decoder_lin2(x)
#         #x = x.view([x.size(0), 256, self.m_finalFrameLeng])
#         x = x.view([x.size(0), 256, -1])
#         x = self.decoder_conv(x)
#         return x, mu, logvar


# # Reconstruction + KL divergence losses summed over all elements and batch
# def vae_loss_function(recon_x, x, mu, logvar,criterion, weight_kld):
#     #BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), size_average=False)
#     loss = criterion(recon_x, x)

#     # see Appendix B from VAE paper:
#     # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
#     # https://arxiv.org/abs/1312.6114
#     # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
#     #KLD = weight_kld * -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
#     KLD = weight_kld * -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

#     return loss +KLD, loss, KLD
#     #return KLD

# '''
# Origial Network Code by Holden
# n=window=240: input data frame length
# d=70: degree of freedom of the body model
# w0=25: temporal filder width
# m=256:number of hidden unites
# batchsize:
# Network(
#         DropoutLayer(amount=dropout, rng=rng),
#         Conv1DLayer(filter_shape=(256, 73, 25), input_shape=(batchsize, 73, window), rng=rng),
#         BiasLayer(shape=(256, 1)),
#         ActivationLayer(),
#         Pool1DLayer(input_shape=(batchsize, 256, window)),
#     ),
    
#     Network(
#         Depool1DLayer(output_shape=(batchsize, 256, window), depooler='random', rng=rng),
#         DropoutLayer(amount=dropout, rng=rng),
#         Conv1DLayer(filter_shape=(73, 256, 25), input_shape=(batchsize, 256, window), rng=rng),
#         BiasLayer(shape=(73, 1))
#     )
# '''


# """Vectorize the latent space"""
# #Input: (batch, 73 dim, 200 frames)
# class autoencoder_vectorize(nn.Module):
#     def __init__(self):
#         super(autoencoder_vectorize, self).__init__()
#         self.encoder_conv = nn.Sequential(
#             #nn.Dropout(0.25),
#             nn.Conv1d(73,256,25,padding=12),
#             nn.ReLU(True),
#             nn.MaxPool1d(kernel_size=2, stride=2), #(batch, 256, 120)  #256*120= 30720
            
#         )
#         self.encoder_lin = nn.Linear(256*120,1024)   

#         self.decoder_lin = nn.Linear(1024,256*120)   
#         self.decoder_conv = nn.Sequential(    
#             #nn.Dropout(0.25),
#             nn.ConvTranspose1d(256, 73, 25, stride=2, padding=12, output_padding=1),
#             #nn.ReLU()
#           )  

#     def forward(self, x):
#         x = self.encoder_conv(x)
#         x = x.view(-1,256*120)
#         x = self.encoder_lin(x)
        
#         x = self.decoder_lin(x)
#         x = x.view([x.size(0), 256, 120])
#         x = self.decoder_conv(x)
#         return x 