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


# import torch.nn.functional as F
# def test(args, model, device, test_loader):
#     model.eval()
#     test_loss = 0
#     correct = 0
#     with torch.no_grad():
#         for data, target in test_loader:
#             data, target = data.to(device), target.to(device)
#             output = model(data)
#             test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
#             pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
#             correct += pred.eq(target.view_as(pred)).sum().item()

#     test_loss /= len(test_loader.dataset)
#     print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#         test_loss, correct, len(test_loader.dataset),
#         100. * correct / len(test_loader.dataset)))

""" Setting """
bShowTestingError= False


#rng = np.random.RandomState(23455)
rng = np.random.RandomState()

dbfolder = '/ssd/codes/pytorch_motionSynth/motionsynth_data/data/processed/'
dbName = 'data_h36m_testing'
dbName = 'data_h36m_training'
dbName = 'data_cmu'


"""Select multiple models"""
#FolderName, epoch to load


checkpointRoot = '/posefs2b/Users/hanbyulj/pytorch_motionSynth/checkpoint/'
#checkpointRoot = './'


folderNames =['autoencoder_3conv_vae_try1',1050,
                'autoencoder_3conv_vae',500,
                'autoencoder_3conv_vae_try3',1000,
                'autoencoder_3conv_vae_64',900]

folderNames =['autoencoder_3conv_vae_try2',7800]    #Panoptic Hagglig

folderNames =['autoencoder_3conv_vae_try3',4900]    #CMU, kld 0.1


folderNames =['autoencoder_3conv_vae_try3',9400,    #CMU, kld 0.1
                'autoencoder_3conv_vae',5700, #HoldenAll, kld 1.0
            'autoencoder_3conv_vae_try2',37850]  #Panoptic Hagglig



folderNames =['autoencoder_3conv_vae_try3',9400,    #CMU, kld 0.1
                'autoencoder_3conv_vae',5700, #HoldenAll, kld 1.0
            'autoencoder_3conv_vae_try2',37850]  #Panoptic Hagglig


# rng = np.random.RandomState(23456)
# torch.manual_seed(23456)
# torch.cuda.manual_seed(23456)

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

    model.load_state_dict(loaded_state,strict=False) #strict False is needed some version issue for batchnorm
    model = model.eval()
    #model.train()

    modelList.append(model)

rawdata = np.load(dbfolder + dbName+ '.npz') #(frames, 240, 73)
X = rawdata['clips']
X = np.swapaxes(X, 1, 2).astype(np.float32) #(17944, 73, 240)
I = np.arange(len(X))
#rng.shuffle(I); X = X[I]
X_stdd = (X - preprocess['Xmean']) / preprocess['Xstd']


sampleNum = 1
for _ in range(100):
    sample_64 = torch.randn(sampleNum, 64)#.to(device)
    sample_512 = torch.randn(sampleNum, 512)#.to(device)
    
    outputList = list()
    for i, model  in enumerate(modelList):

        """Original"""
        if model.latent_size()==64:
            sample = sample_64
        else:
            sample = sample_512
        Xrecn = model.decode(sample)
        #Xrecn = (Xrecn.data.numpy() * preprocess['Xstd']) + preprocess['Xmean']
        Xrecn = (Xrecn.data.numpy() * preprocess['Xstd']) + preprocess['Xmean']
        
        #Xrecn[:,-7:-4] = Xorgi[:,-7:-4] #Align Centers for Debug

        for i in range(Xrecn.shape[0]):
            outputList.append(Xrecn[i,:,:])

    show_Holden_Data_73(outputList)
        

