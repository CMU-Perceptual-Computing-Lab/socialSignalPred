import sys
import numpy as np
import time

from torch.autograd import Variable

#by jhugestar
sys.path.append('/ssd/codes/glvis_python/')
from Visualize_human_gl import showSkeleton,show_Holden_Data_73 #opengl visualization 

# skels = np.load('denoise_out.npy')
# showSkeleton([skels[0], skels[1]])
# showSkeleton([skels[1], skels[2]])

# t = threading.Thread(target=showSkeleton, args=([skels[1], skels[2]],))
# #threads.append(t)
# t.start()
# print('hello')
# t.join()

# sys.path.append('../nn')
# from AnimationPlot import animation_plot
# animation_plot( [ skels[0], skels[1], skels[2]], interval=2)    
# import scipy.io as io
# import theano
# import theano.tensor as T
# from network import create_core
#from constraints import constrain, foot_sliding, joint_lengths, trajectory, multiconstraint

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


model.eval()

rng = np.random.RandomState(23455)

# import matplotlib.pyplot as plt
# plt.plot([1,2,3,4])
# plt.ylabel('some numbers')
# plt.show()

#X = np.load('../data/processed/data_edin_locomotion.npz')['clips']
#X = np.load('../data/processed/data_hdm05.npz')['clips']
#X = np.load('/ssd/codes/pytorch_motionSynth/motionsynth_data/data/processed/data_cmu.npz')['clips'] #(17944, 240, 73)
#X = np.load('/ssd/codes/pytorch_motionSynth/motionsynth_data/data/processed/data_h36m_training.npz')['clips'] #(17944, 240, 73)
#X = np.load('/ssd/codes/pytorch_motionSynth/motionsynth_data/data/processed/data_h36m_testing.npz')['clips'] #(17944, 240, 73)

X = np.load('/ssd/codes/pytorch_motionSynth/motionsynth_data/data/processed/data_panoptic_haggling_all.npz')['clips'] #(17944, 240, 73)
X = np.swapaxes(X, 1, 2).astype(np.float32) #(17944, 73, 240)

preprocess = np.load('preprocess_core.npz') #preprocess['Xmean' or 'Xstd']: (1, 73,1)
X = (X - preprocess['Xmean']) / preprocess['Xstd']

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

    # """Visualizing original data only"""
    # Xorgi = (Xorgi * preprocess['Xstd']) + preprocess['Xmean']
    # show_Holden_Data_73([ Xorgi[0,:,:]])
    # continue

    """Gaussian Noise"""
    #Xnois = ((Xorgi * rng.binomial(size=Xorgi.shape, n=1, p=0.5)) / 0.5).astype(np.float32)

    """Missing Noise"""
    # Xnois = Xorgi.copy()#((Xorgi * rng.binomial(size=Xorgi.shape, n=1, p=0.5)) / 0.5).astype(np.float32)
    # Xnois[:,:,::5] = 0

    #inputData = Variable(torch.from_numpy(inputData)).cuda()
    Xrecn = model(Variable(torch.from_numpy(Xorgi)))    #on CPU 
    #Xrecn = model(Variable(torch.from_numpy(Xnois)))    #on CPU 
    #Xrecn = np.array(network(Xnois).eval())    

    Xorgi = (Xorgi * preprocess['Xstd']) + preprocess['Xmean']
    #Xnois = (Xnois * preprocess['Xstd']) + preprocess['Xmean']
    Xrecn = (Xrecn.data.numpy() * preprocess['Xstd']) + preprocess['Xmean']

    # Xrecn = constrain(Xrecn, network[0], network[1], preprocess, multiconstraint(
    #     foot_sliding(Xorgi[:,-4:].copy()),
    #     joint_lengths(),
    #     trajectory(Xorgi[:,-7:-4])), alpha=0.01, iterations=50)

    Xrecn[:,-7:-4] = Xorgi[:,-7:-4]

    #animation_plot([Xnois, Xrecn, Xorgi], interval=2)    
    #animation_plot([Xnois, Xrecn, Xorgi], interval=15.15)
    #show_Holden_Data_73([ Xrecn[0,:,:], Xorgi[0,:,:], Xnois[0,:,:] ])
    #show_Holden_Data_73([ Xrecn[0,:,:], Xorgi[0,:,:], Xnois[0,:,:] ])
    show_Holden_Data_73([ Xrecn[0,:,:], Xorgi[0,:,:] ])