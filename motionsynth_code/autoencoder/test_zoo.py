import sys
import numpy as np
import time
import modelZoo
import torch
from torch.autograd import Variable

#by jhugestar
sys.path.append('/ssd/codes/glvis_python/')
from Visualize_human_gl import showSkeleton,show_Holden_Data_73 #opengl visualization 


#checkpointFolder = '/posefs2b/Users/hanbyulj/pytorch_motionSynth/checkpoint/'
checkpointFolder = './'
runName = '/autoencoder_2convLayers/'
preprocess = np.load(checkpointFolder+ runName + 'preprocess_core.npz') #preprocess['Xmean' or 'Xstd']: (1, 73,1)

#model = modelZoo.autoencoder_first()
#model = modelZoo.autoencoder_vectorize()
model = modelZoo.autoencoder_2convLayers()
#model = modelZoo.autoencoder_3convLayers()
#trainResultName = checkpointFolder+ runName + 'motion_autoencoder_naive_dropout_h36mOnly_dropout390.pth'
trainResultName = checkpointFolder+ runName + 'checkpoint_e10.pth'
model.load_state_dict(torch.load(trainResultName, map_location=lambda storage, loc: storage))
model.eval()

rng = np.random.RandomState(23455)
dbfolder = '/ssd/codes/pytorch_motionSynth/motionsynth_data/data/processed/'
dbName = 'data_h36m_testing'
dbName = 'data_h36m_training'

rawdata = np.load(dbfolder + dbName+ '.npz') #(frames, 240, 73)
X = rawdata['clips']
X = np.swapaxes(X, 1, 2).astype(np.float32) #(17944, 73, 240)
X_stdd = (X - preprocess['Xmean']) / preprocess['Xstd']


batchsize = 1
window = X.shape[2]

for _ in range(100):
    index = rng.randint(X.shape[0])
    print(index)
    Xorgi = X[index:index+1,:,:]
    Xorgi_stdd = X_stdd[index:index+1,:,:]  #Input (batchSize,73,240) 

    #show_Holden_Data_73([ Xorgi[0,:,:]])

    """Original"""
    #inputData = Variable(torch.from_numpy(inputData)).cuda()
    Xrecn = model(Variable(torch.from_numpy(Xorgi_stdd)))
    Xrecn = (Xrecn.data.numpy() * preprocess['Xstd']) + preprocess['Xmean']
    
    Xrecn[:,-7:-4] = Xorgi[:,-7:-4] #Align Centers for Debug

    show_Holden_Data_73([ Xorgi[0,:,:], Xrecn[0,:,:]])
    #show_Holden_Data_73([ Xorgi[0,:,:], Xrecn[0,:,:], Xrecn_winner[0,:,:]])