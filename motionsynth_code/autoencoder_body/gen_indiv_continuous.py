import os
import sys
import numpy as np
import scipy.io as io
import random

import torch
import torchvision
from torch import nn
from torch.autograd import Variable
import os

import modelZoo

# Utility Functions
from utility import print_options,save_options
from utility import setCheckPointFolder
from utility import my_args_parser

#by jhugestar
sys.path.append('/ssd/codes/glvis_python/')
#from glViewer import SetFaceParmData,setSpeech,setSpeechGT,setSpeech_binary, setSpeechGT_binary, init_gl #opengl visualization 
import glViewer

######################################3
# Logging
import logging
#FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
FORMAT = '[%(levelname)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)  ##default logger


######################################3
# Parameter Handling
parser = my_args_parser()
args = parser.parse_args()

# Some initializations #
torch.cuda.set_device(args.gpu)

rng = np.random.RandomState(23456)
torch.manual_seed(23456)
torch.cuda.manual_seed(23456)


######################################
# Dataset 
#datapath ='/ssd/codes/pytorch_motionSynth/motionsynth_data' 
datapath ='../../motionsynth_data/data/processed/' 

#test_dblist = ['data_hagglingSellers_speech_face_60frm_10gap_white_testing']
# test_dblist = ['data_hagglingSellers_speech_body_120frm_10gap_white_testing']

# test_data = np.load(datapath + test_dblist[0] + '.npz')
# test_X_raw = test_data['clips']  #Input (1044,240,73)
# test_speech_raw  = test_data['speech']  #Input (1044,240,73)

# test_X = test_X[:100,:,:]
# test_Y = test_Y[:100]

######################################
# Checkout Folder and pretrain file setting
# checkpointRoot = './'
# checkpointFolder = checkpointRoot+ '/social_autoencoder_3conv_vect_vae/'
# preTrainFileName= 'checkpoint_e50_loss0.0527.pth'

# checkpointRoot = '/posefs2b/Users/hanbyulj/pytorch_motionSynth/checkpoint/'
# checkpointFolder = checkpointRoot+ '/3_body_kldWeight/social_autoencoder_3conv_vect_vae_speakOnly_kld0.001/'
# preTrainFileName= 'checkpoint_e3850_loss0.6438.pth'

checkpointRoot = './'
checkpointFolder = checkpointRoot+ 'social_autoencoder_3conv_vect_vae_try13/'
preTrainFileName= 'checkpoint_e41_loss0.3137.pth'


######################################
# Input/Output Option
# test_X = test_X_raw#[1,:,:,:]      #(num, chunkLength, dim:200) //person 1 (first seller's value)
#test_Y = test_X_raw[2,:,:,:]

######################################
# Data pre-processing
# test_X = np.swapaxes(test_X, 1, 2).astype(np.float32) #(num, frames, dim:200) => (num, 200, frames)
#test_Y = np.swapaxes(test_Y, 1, 2).astype(np.float32) #(num, frames, dim:200) => (num, 200, frames)
#train_Y = train_Y.astype(np.float32)



######################################
# Data pre-processing
preprocess = np.load(checkpointFolder + 'preprocess_core.npz') #preprocess['Xmean' or 'Xstd']: (1, 73,1)

Xmean = preprocess['Xmean']
Xstd = preprocess['Xstd']

#Ymean = preprocess['Ymean']
#Ystd = preprocess['Ystd']

# test_X_std = (test_X - Xmean) / Xstd
#test_Y_std = (test_Y - Xmean) / Xstd


######################################
# Network Setting
#batch_size = args.batch
batch_size = 1#512
#criterion = nn.BCELoss()
criterion = nn.MSELoss()

#Creat Model
#model = modelZoo.autoencoder_first().cuda()
# featureDim = test_X_raw.shape[2]
featureDim = 73
latentDim = 200
model = modelZoo.autoencoder_3conv_vect_vae(featureDim,latentDim).cuda()
model.eval()

#Creat Model
trainResultName = checkpointFolder + preTrainFileName
loaded_state = torch.load(trainResultName, map_location=lambda storage, loc: storage)

model.load_state_dict(loaded_state,strict=False) #strict False is needed some version issue for batchnorm
model = model.eval()  #Do I need this again?



# ######################################
# # Generation
# sampleNum = 2
# for _ in range(100):
#     sample_2 = torch.randn(sampleNum, 200)#.to(device)

#     print(sample_2[:10])
    
#     sample_2 = Variable(sample_2).cuda()
#     output = model.decode(sample_2)


#     #De-standardaize
#     output_np = output.data.cpu().numpy()  #(batch, featureDim, frames)
#     output_np = output_np*Xstd + Xmean

#     output_np = np.swapaxes(output_np,1,2)  #(batch, frames, featureDim)
#     output_np = np.reshape(output_np,(-1,featureDim))
#     output_np = np.swapaxes(output_np,0,1)  #(featureDim, frames)

#     bodyData = [output_np]
#     glViewer.set_Holden_Data_73(bodyData)
#     glViewer.init_gl()


######################################
# Continuout Motion Generation
sampleNum = 2
for _ in range(100):

    output_list =[]
    prev_encode = None
    for iter in range(sampleNum):
    
        if prev_encode is None:
            new_encode = torch.randn(1, 200)#.to(device)
        else:
            new_encode = prev_encode + torch.randn(1, 200)*0.2#.to(device)
    
        new_encode_cuda = Variable(new_encode).cuda()
        output = model.decode(new_encode_cuda)

        #De-standardaize
        output_np = output.data.cpu().numpy()  #(batch, featureDim, frames)
        output_np = output_np*Xstd + Xmean

        output_np = np.swapaxes(output_np,1,2)  #(batch, frames, featureDim)
        output_np = np.reshape(output_np,(-1,featureDim))
        output_np = np.swapaxes(output_np,0,1)  #(featureDim, frames)
        
        prev_encode = new_encode
        output_list.append(output_np)


    output_np =  np.concatenate(output_list,axis=1)
    bodyData = [output_np]
    glViewer.set_Holden_Data_73(bodyData)
    glViewer.init_gl()
