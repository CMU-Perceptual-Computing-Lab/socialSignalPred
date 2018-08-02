import sys
import numpy as np
import time

from torch.autograd import Variable

#by jhugestar
sys.path.append('/ssd/codes/glvis_python/')
from Visualize_human_gl import showSkeleton,show_Holden_Data_73 #opengl visualization 

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


preprocess = np.load('/media/posefs2b/Users/hanbyulj/pytorch_motionSynth/checkpoint/autoencoder/preprocess_core.npz') #preprocess['Xmean' or 'Xstd']: (1, 73,1)

model = autoencoder()
model.load_state_dict(torch.load('/media/posefs2b/Users/hanbyulj/pytorch_motionSynth/checkpoint/autoencoder/motion_autoencoder_naive_dropout_h36mOnly_dropout500.pth', map_location=lambda storage, loc: storage))
model.eval()

model_winner = autoencoder()
model_winner.load_state_dict(torch.load('haggling_winner_fineTune499.pth', map_location=lambda storage, loc: storage))
model_winner.eval()

rng = np.random.RandomState(23455)
dbfolder = './panoptic_cleaned/'

# dbNames = ('data_panoptic_haggling_all',
#                  'data_panoptic_haggling_buyers',
#                  'data_panoptic_haggling_sellers',
#                  'data_panoptic_haggling_losers',
#                  'data_panoptic_haggling_winners'
#                   )
#dbName = 'data_panoptic_haggling_sellers'
dbName = 'data_panoptic_haggling_buyers'
rawdata = np.load(dbfolder + dbName+ '.npz') #(frames, 240, 73)

X = rawdata['clips']
X = np.swapaxes(X, 1, 2).astype(np.float32) #(17944, 73, 240)
X_stdd = (X - preprocess['Xmean']) / preprocess['Xstd']


batchsize = 1
window = X.shape[2]

# 2021
# 1
#3283
#with torch.no_grad():
for _ in range(100):
    index = rng.randint(X.shape[0])
    print(index)
    Xorgi = X[index:index+1,:,:]
    Xorgi_stdd = X_stdd[index:index+1,:,:]  #Input (batchSize,73,240) 

    """Original"""
    #inputData = Variable(torch.from_numpy(inputData)).cuda()
    Xrecn = model(Variable(torch.from_numpy(Xorgi_stdd)))    #on CPU 
    Xrecn = (Xrecn.data.numpy() * preprocess['Xstd']) + preprocess['Xmean']
    Xrecn[:,-7:-4] = Xorgi[:,-7:-4]

    """Adding gaussian noise on latent space"""
    Xrecn_latentNoisy = model.encoder(Variable(torch.from_numpy(Xorgi_stdd)))   #Xrecn_latentNoisy:  [1, 256, 120 ]
    noise = np.random.normal(scale=0.5, size=list(Xrecn_latentNoisy.size())).astype(np.float32)
    #noise = rng.binomial(size=list(Xrecn_latentNoisy.size()), n=1, p=0.5)) / 0.5).astype(np.float32)
    print('noisemax: {0}'.format(max(noise.flatten())))
    #Xrecn_again = model.decoder(Xrecn_latentNoisy + Variable(torch.from_numpy(noise)))
    Xrecn_synth = model.decoder(Variable(torch.from_numpy(noise)))
    Xrecn_synth = (Xrecn_synth.data.numpy() * preprocess['Xstd']) + preprocess['Xmean']
    #Xrecn_again[:,-7:-4] = Xorgi[:,-7:-4]

    # """Haggling Winner Finetuned"""
    # Xrecn_winner = model_winner(Variable(torch.from_numpy(Xorgi_stdd)))    #on CPU 
    # Xrecn_winner = (Xrecn_winner.data.numpy() * preprocess['Xstd']) + preprocess['Xmean']
    # Xrecn_winner[:,-7:-4] = Xorgi[:,-7:-4]

    
    #rng.binomial(size=list(Xrecn_latentNoisy.size()), n=1, p=0.5)) / 0.5).astype(np.float32)
    #rng.binomial(size=list(Xrecn_latentNoisy.size()), n=1, p=0.5)) / 0.5).astype(np.float32)
    

    #show_Holden_Data_73([ Xorgi[0,:,:], Xrecn[0,:,:], Xrecn_again[0,:,:]])
    #show_Holden_Data_73([ Xorgi[0,:,:], Xrecn_again[0,:,:]])
    show_Holden_Data_73([ Xrecn_synth[0,:,:]])
    #show_Holden_Data_73([ Xorgi[0,:,:], Xrecn[0,:,:], Xrecn_winner[0,:,:]])