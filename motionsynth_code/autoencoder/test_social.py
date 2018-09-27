import sys
import numpy as np
import time
import modelZoo
import torch
from torch.autograd import Variable
import json
import os


'''patch for loading pytorch 0.4 pkl files'''
import torch._utils
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2


#by jhugestar
sys.path.append('/ssd/codes/glvis_python/')
from Visualize_human_gl import showSkeleton,show_Holden_Data_73 #opengl visualization 


""" Setting """
bShowTestingError= True


#rng = np.random.RandomState(23455)
rng = np.random.RandomState()

dbfolder = '/ssd/codes/pytorch_motionSynth/motionsynth_data/data/processed/'

# dbName = 'data_h36m_training'
# dbName = 'data_panoptic_haggling_testing'
# dbName = 'data_mhad'
# dbName = 'data_h36m_testing'
# dbName = 'data_panoptic_haggling_sellers'
# dbName = 'data_cmu'
# dbName_input = 'data_panoptic_haggling_winners'
# dbName_output = 'data_panoptic_haggling_losers'

dbName_output = 'data_panoptic_haggling_winners'
dbName_input = 'data_panoptic_haggling_losers'


"""Select multiple models"""
#FolderName, epoch to load

checkpointRoot = '/posefs2b/Users/hanbyulj/pytorch_motionSynth/checkpoint/'
#checkpointRoot = checkpointRoot + 'test_Ae_varyingLatentDim/'
#checkpointRoot = './'

folderNames =['autoencoder_3conv_vae_try6',500,
                'autoencoder_3conv_vae_try5',1850]

folderNames =['autoencoder_3conv_vect3_8',3300,
                '1_autoencoder_3conv_vae_mean',500,
                'autoencoder_3conv_vae_try1',700,
                'autoencoder_3conv_vae_try3',700]

folderNames =['autoencoder_3conv_vae_try2',7800]

#Latent Vector Test (1024, )
# folderNames =['autoencoder_3conv_vect3_2',24250,
#         'autoencoder_3conv_vect3_8',6200,
#         'autoencoder_3conv_vect3_64',4450]

# folderNames =['autoencoder_3conv_vect3_2',24250,
#          'autoencoder_3conv_vect3_8',6200,
#          'autoencoder_3conv_vect3_64',4450]

folderNames =['autoencoder_3conv_vae_kld0.1', 17100,
        'social_autoencoder_3convLayers_vect',20500,
        'social_autoencoder_3conv_vae_try1',32150]

rng = np.random.RandomState(23456)
torch.manual_seed(23456)
torch.cuda.manual_seed(23456)

from torch import nn
criterion = nn.MSELoss() 
#nn.MSELoss(size_average=False) #Sum

modelList = list()
for folderName, loadEpoch in zip(folderNames[0::2],folderNames[1::2]):

    checkpointFolder = checkpointRoot+folderName + '/'
    #load log file
    log_file_name = os.path.join(checkpointFolder, 'opt.json')
    with open(log_file_name, 'r') as opt_file:
        options_dict = json.load(opt_file)

    #model = modelZoo.autoencoder_first()
    #loadEpoch = 1000
    model = getattr(modelZoo,options_dict['model'])()

    #runFolderName = model.__class__.__name__ + '/'#'/autoencoder_3convLayers/'
    preprocess = np.load(checkpointFolder + 'preprocess_core.npz') #preprocess['Xmean' or 'Xstd']: (1, 73,1)

    #trainResultName = checkpointFolder+ runName + 'motion_autoencoder_naive_dropout_h36mOnly_dropout390.pth'
    trainResultName = checkpointFolder + 'checkpoint_e' + str(loadEpoch) + '.pth'


    loaded_state = torch.load(trainResultName, map_location=lambda storage, loc: storage)

    #Some version issue..
    #for key in loaded_state.keys():
    #    if 'num_batches' in key:
    #        loaded_state.pop(key,None)

    #model.load_state_dict(loaded_state) 
    model.load_state_dict(loaded_state,strict=False) #strict False is needed some version issue for batchnorm
    model = model.eval()
    #model.train()
    #model.train()

    modelList.append(model)



rawdata = np.load(dbfolder + dbName_input+ '.npz') #(frames, 240, 73)
X = rawdata['clips']
X = np.swapaxes(X, 1, 2).astype(np.float32) #(17944, 73, 240)
#I = np.arange(len(X))
#rng.shuffle(I); X = X[I]
X_stdd = (X - preprocess['Xmean']) / preprocess['Xstd']

rawdata = np.load(dbfolder + dbName_output+ '.npz') #(frames, 240, 73)
Y = rawdata['clips']
Y = np.swapaxes(Y, 1, 2).astype(np.float32) #(17944, 73, 240)
#I = np.arange(len(X))
#rng.shuffle(I); X = X[I]
Y_stdd = (Y - preprocess['Xmean']) / preprocess['Xstd']



if bShowTestingError:
    batch_size = 1024
    """Compute Testing Errors"""
    batchinds = np.arange(X.shape[0] // batch_size)
    for i, model  in enumerate(modelList):
        model_gpu = model.cuda()
        test_loss = 0
        for bii, bi in enumerate(batchinds):
        # print('{} /{}\n'.format(bii,len(batchinds)))
            idxStart  = bi*batch_size
            Xorgi_stdd = X_stdd[idxStart:(idxStart+batch_size),:,:]  #Input (batchSize,73,240) 
            Yorgi_stdd = Y_stdd[idxStart:(idxStart+batch_size),:,:]  #Input (batchSize,73,240) 
            Xrecn = model_gpu(Variable(torch.from_numpy(Xorgi_stdd)).cuda())
            if len(Xrecn)==3:
                Xrecn = Xrecn[0]
            loss = criterion(Xrecn, Variable(torch.from_numpy(Yorgi_stdd)).cuda())
            test_loss += loss.data.cpu().numpy().item()* batch_size # sum up batch loss

        model.cpu() #Put model back to CPU
        test_loss /= len(batchinds)*batch_size
        print('Testing: modelName: {} (epoch:{}) /Average loss: {:.4f}\n'.format(folderNames[2*i],folderNames[2*i+1],test_loss))
        #     test_loss, correct, len(test_loader.dataset),
        #     100. * correct / len(test_loader.dataset)))


for _ in range(100):
    index = rng.randint(X.shape[0])
#for index in [ 7895,   604, 15619,  8415, 16100,  1677,  2151, 16152, 10973, 17592]:
    print("frame: {}".format(index))
    Xorgi = X[index:index+1,:,:]
    Xorgi_stdd = X_stdd[index:index+1,:,:]  #Input (batchSize,73,240) 

    Yorgi = Y[index:index+1,:,:]
    Yorgi_stdd = Y_stdd[index:index+1,:,:]  #Input (batchSize,73,240) 

    outputList = list()
    outputList.append(Xorgi[0,:,:])
    outputList.append(Yorgi[0,:,:])
    for i, model  in enumerate(modelList):

        """Original"""
        #inputData = Variable(torch.from_numpy(inputData)).cuda()
        Xrecn = model(Variable(torch.from_numpy(Xorgi_stdd)))
        if len(Xrecn)>1:
                Xrecn = Xrecn[0]
        #loss = criterion(Xrecn, Variable(torch.from_numpy(Xorgi_stdd)))
        loss = criterion(Xrecn, Variable(torch.from_numpy(Yorgi_stdd)))
        print("model {0}: {1:.4f}".format(i, loss.item()))
        Xrecn = (Xrecn.data.numpy() * preprocess['Xstd']) + preprocess['Xmean']
        
        #Xrecn[:,-7:-4] = Xorgi[:,-7:-4] #Align Centers for Debug

        outputList.append(Xrecn[0,:,:])

    show_Holden_Data_73(outputList)
    #show_Holden_Data_73([ Xorgi[0,:,:], Xrecn[0,:,:]])
    #show_Holden_Data_73([ Xorgi[0,:,:], Xrecn[0,:,:], Xrecn_winner[0,:,:]])