import sys
import numpy as np
import time
import os

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


model = autoencoder()
#model.load_state_dict(torch.load('./motion_autoencoder_naive.pth', map_location=lambda storage, loc: storage))
#model.load_state_dict(torch.load('./motion_autoencoder_naive_dropout.pth', map_location=lambda storage, loc: storage))
#model.load_state_dict(torch.load('./motion_autoencoder_naive_dropout_2nd.pth', map_location=lambda storage, loc: storage))
#model.load_state_dict(torch.load('./motion_autoencoder_naive_dropout_h36mOnly.pth', map_location=lambda storage, loc: storage))
model.load_state_dict(torch.load('/media/posefs2b/Users/hanbyulj/pytorch_motionSynth/checkpoint/autoencoder/motion_autoencoder_naive_dropout_h36mOnly_dropout500.pth', map_location=lambda storage, loc: storage))
preprocess = np.load('/media/posefs2b/Users/hanbyulj/pytorch_motionSynth/checkpoint/autoencoder/preprocess_core.npz') #preprocess['Xmean' or 'Xstd']: (1, 73,1)

outputFolder = './panoptic_cleaned/'
if not os.path.exists(outputFolder):
    os.mkdir(outputFolder)

model.eval()

rng = np.random.RandomState(23455)

dbNames = ('data_panoptic_haggling_all',
                 'data_panoptic_haggling_buyers',
                 'data_panoptic_haggling_sellers',
                 'data_panoptic_haggling_losers',
                 'data_panoptic_haggling_winners'
                  )
# X = np.load(dbfolder + 'data_panoptic_haggling_all.npz') #(8049, 240, 73)
#rawdata = np.load(dbfolder + 'data_panoptic_haggling_buyers.npz') #(2683, 240, 73)
# X = np.load(dbfolder + 'data_panoptic_haggling_sellers.npz')['clips'] #(5366, 240, 73)
#X = np.load(dbfolder + 'data_panoptic_haggling_losers.npz')['clips'] #(2683, 240, 73)
#X = np.load(dbfolder + 'data_panoptic_haggling_winners.npz')['clips'] #(2683, 240, 73)

dbfolder = '/ssd/codes/pytorch_motionSynth/motionsynth_data/data/processed/'

for dbName in dbNames:

    rawdata = np.load(dbfolder + dbName+ '.npz') #(frames, 240, 73)

    X = rawdata['clips']
    X = np.swapaxes(X, 1, 2).astype(np.float32) #(17944, 73, 240)

    X_stdd = (X - preprocess['Xmean']) / preprocess['Xstd']

    batchsize = 1
    window = X.shape[2]

    for index in range(X.shape[0]):
    #for _ in range(100):
        #index = rng.randint(X.shape[0])
        
        print(index)
        Xorgi = X[index:index+1,:,:]
        Xorgi_stdd = X_stdd[index:index+1,:,:]

        # # """Visualizing original data only"""
        #show_Holden_Data_73([ Xorgi[0,:,:]])
        #Continue

        #inputData = Variable(torch.from_numpy(inputData)).cuda()
        Xrecn = model(Variable(torch.from_numpy(Xorgi_stdd)))    
        #Xrecn_first = (Xrecn.data.numpy() * preprocess['Xstd']) + preprocess['Xmean']
        #Xrecn = model(Variable(torch.from_numpy(Xrecn.data.numpy())))    
        #Xrecn = model(Variable(torch.from_numpy(Xnois)))    #on CPU 
        #Xrecn = np.array(network(Xnois).eval())    

        Xrecn = (Xrecn.data.numpy() * preprocess['Xstd']) + preprocess['Xmean']
        Xrecn[:,-7:-4] = Xorgi[:,-7:-4]

        #Save
        X[index:index+1,:,:] = Xrecn
        #show_Holden_Data_73([ Xorgi[0,:,:], Xrecn[0,:,:]])

    X = np.swapaxes(X, 1, 2).astype(np.float64) #(frames, 73, 240) ->(frames, 240, 73)
    #X = np.load(dbfolder + 'data_panoptic_haggling_buyers.npz')['clips'] #(2683, 240, 73)
    #preprocess = np.save('/media/posefs2b/Users/hanbyulj/pytorch_motionSynth/checkpoint/autoencoder/preprocess_core.npz') #preprocess['Xmean' or 'Xstd']: (1, 73,1)
    np.savez_compressed(outputFolder + dbName, clips=X, classes=rawdata['classes'])