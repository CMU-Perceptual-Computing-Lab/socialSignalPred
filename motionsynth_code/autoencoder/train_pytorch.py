
import os
import sys
import numpy as np
import scipy.io as io
#import theano
#import theano.tensor as T
#sys.path.append('../nn')
#from AdamTrainer import AdamTrainer
#from AnimationPlot import animation_plot
#from network import create_core

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

datapath ='/ssd/codes/pytorch_motionSynth/motionsynth_data' 
Xcmu = np.load(datapath +'/data/processed/data_cmu.npz')['clips'] # (17944, 240, 73)
Xhdm05 = np.load(datapath +'/data/processed/data_hdm05.npz')['clips']	#(3190, 240, 73)
Xmhad = np.load(datapath +'/data/processed/data_mhad.npz')['clips'] # (2674, 240, 73)
#Xstyletransfer = np.load('/data/processed/data_styletransfer.npz')['clips']
Xedin_locomotion = np.load(datapath +'/data/processed/data_edin_locomotion.npz')['clips'] #(351, 240, 73)
Xedin_xsens = np.load(datapath +'/data/processed/data_edin_xsens.npz')['clips'] #(1399, 240, 73)
Xedin_misc = np.load(datapath +'/data/processed/data_edin_misc.npz')['clips'] #(122, 240, 73)
Xedin_punching = np.load(datapath +'/data/processed/data_edin_punching.npz')['clips'] #(408, 240, 73)
h36m_training = np.load(datapath +'/data/processed/data_h36m_training.npz')['clips'] #(13156, 240, 73)


#X = np.concatenate([Xcmu, Xhdm05, Xmhad, Xstyletransfer, Xedin_locomotion, Xedin_xsens, Xedin_misc, Xedin_punching], axis=0)
#X = np.concatenate([Xcmu, Xhdm05, Xmhad, Xedin_locomotion, Xedin_xsens, Xedin_misc, Xedin_punching], axis=0) #(26088,  240, 73 )
X = h36m_training
#X = np.swapaxes(X, 1, 2).astype(theano.config.floatX) #(26088,  73,  240)
X = np.swapaxes(X, 1, 2).astype(np.float32)

feet = np.array([12,13,14,15,16,17,24,25,26,27,28,29])

Xmean = X.mean(axis=2).mean(axis=0)[np.newaxis,:,np.newaxis]
Xmean[:,-7:-4] = 0.0
Xmean[:,-4:]   = 0.5

Xstd = np.array([[[X.std()]]]).repeat(X.shape[1], axis=1)
Xstd[:,feet]  = 0.9 * Xstd[:,feet]
Xstd[:,-7:-5] = 0.9 * X[:,-7:-5].std()
Xstd[:,-5:-4] = 0.9 * X[:,-5:-4].std()
Xstd[:,-4:]   = 0.5

#np.savez_compressed('preprocess_core.npz', Xmean=Xmean, Xstd=Xstd)

X = (X - Xmean) / Xstd

I = np.arange(len(X))
rng.shuffle(I); X = X[I]

print(X.shape)

# E = theano.shared(X, borrow=True)
# batchsize = 1
#network = create_core(rng=rng, batchsize=batchsize, window=X.shape[2])
#trainer = AdamTrainer(rng=rng, batchsize=batchsize, epochs=100, alpha=0.00001)
#trainer.train(network, E, E, filename='network_core.npz')


import torch
from torch import nn
class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Dropout(0.25),
            nn.Conv1d(73,256,25,padding=12),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=2, stride=2)
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

num_epochs = 500
batch_size = 128
learning_rate = 1e-3

model = autoencoder().cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

for epoch in range(num_epochs):

    batchinds = np.arange(X.shape[0] // batch_size)
    rng.shuffle(batchinds)
    
    for bii, bi in enumerate(batchinds):
        inputData = X[bi:(bi+batch_size),:,:]
        inputData = Variable(torch.from_numpy(inputData)).cuda()

        # ===================forward=====================
        output = model(inputData)
        loss = criterion(output, inputData)

        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================log========================
        print('epoch [{}/{}], loss:{:.4f}'
                    .format(epoch + 1, num_epochs, loss.data[0]))

    # for data in dataloader:
    #     img, _ = data
    #     img = img.view(img.size(0), -1)
    #     img = Variable(img).cuda()q
    #     # ===================forward=====================
    #     output = model(img)
    #     loss = criterion(output, img)
    #     # ===================backward====================
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()
    # # ===================log========================
    # print('epoch [{}/{}], loss:{:.4f}'
    #       .format(epoch + 1, num_epochs, loss.data[0]))
    # if epoch % 10 == 0:
    #     pic = to_img(output.cpu().data)
    #     save_image(pic, './mlp_img/image_{}.png'.format(epoch))
    if epoch % 100 == 0:
        fileName = './motion_autoencoder_naive_dropout_h36mOnly_dropout' + str(epoch) + '.pth'
        torch.save(model.state_dict(), fileName)
