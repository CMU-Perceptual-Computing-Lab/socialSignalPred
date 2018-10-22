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
test_dblist = ['data_hagglingSellers_speech_face_120frm_10gap_white_testing']

test_data = np.load(datapath + test_dblist[0] + '.npz')
test_X_raw = test_data['clips']  #Input (1044,240,73)
test_Y_raw  = test_data['speech']  #Input (1044,240,73)

# test_X = test_X[:100,:,:]
# test_Y = test_Y[:100]

######################################
# Checkout Folder and pretrain file setting
#checkpointRoot = './'

train_data=[]
checkpointRoot = '/posefs2b/Users/hanbyulj/pytorch_motionSynth/checkpoint/'


# #kld 0
# checkpointFolder = './save/social_autoencoder_3conv_vect_vae_noKld_latent100/'
# preTrainFileName= 'checkpoint_e2800_loss0.0211.pth'
# train_data.append({'dir':checkpointFolder, 'file':preTrainFileName})


#kld 0.01
checkpointFolder = 'social_autoencoder_3conv_vect_vae_try3/'
preTrainFileName= 'checkpoint_e9150_loss0.0340.pth'
train_data.append({'dir':checkpointFolder, 'file':preTrainFileName})


#kld 0.1
checkpointFolder = checkpointRoot+ '/social_autoencoder_3conv_vect_vae_try1/'
preTrainFileName= 'checkpoint_e4700_loss0.0760.pth'
train_data.append({'dir':checkpointFolder, 'file':preTrainFileName})


# #kld 0.5
# checkpointFolder = checkpointRoot+ '/social_autoencoder_3conv_vect_vae_try2/'
# preTrainFileName= 'checkpoint_e4800_loss0.1377.pth'
# train_data.append({'dir':checkpointFolder, 'file':preTrainFileName})




######################################
# Feature
featureDim = 5
test_X_raw = test_X_raw[:,:,:featureDim]


# ######################################
# # Input/Output Option
# test_X = test_X_raw[1,:,:,:]      #(num, chunkLength, dim:200) //person 1 (first seller's value)
# test_Y = test_X_raw[2,:,:,:]
test_X = test_X_raw


######################################
# Data pre-processing
test_X = np.swapaxes(test_X, 1, 2).astype(np.float32) #(num, frames, dim:200) => (num, 200, frames)
#test_Y = np.swapaxes(test_Y, 1, 2).astype(np.float32) #(num, frames, dim:200) => (num, 200, frames)
#train_Y = train_Y.astype(np.float32)



######################################
# Data pre-processing
preprocess = np.load(checkpointFolder + 'preprocess_core.npz') #preprocess['Xmean' or 'Xstd']: (1, 73,1)

Xmean = preprocess['Xmean']
Xstd = preprocess['Xstd']

#Ymean = preprocess['Ymean']
#Ystd = preprocess['Ystd']

test_X_std = (test_X - Xmean) / Xstd
#test_Y_std = (test_Y - Xmean) / Xstd



######################################
# Network Setting
#batch_size = args.batch
batch_size = 1#512
#criterion = nn.BCELoss()
criterion = nn.MSELoss()

#Creat Model
#model = modelZoo.autoencoder_first(featureDim).cuda()
#model = modelZoo.autoencoder_1conv_vect_vae(featureDim).cuda()


model_list=[]


for train_d  in train_data:
    model = modelZoo.autoencoder_3conv_vect_vae(featureDim, 100).cuda()
    model.eval()

    #Creat Model
    trainResultName = train_d['dir'] + train_d['file']
    loaded_state = torch.load(trainResultName, map_location=lambda storage, loc: storage)

    model.load_state_dict(loaded_state,strict=False) #strict False is needed some version issue for batchnorm
    model = model.eval()  #Do I need this again?
    model_list.append(model)



# Ystd = np.swapaxes(Ystd,1,2)
# Ymean = np.swapaxes(Ymean,1,2)

######################################
# Training
batchinds = np.arange(test_X.shape[0] // batch_size)
pred_all = np.empty([0,1],dtype=float)
test_X_vis =np.empty([0,200],dtype=float)


sampleNum = 1
for _ in range(100):
    sample_2 = torch.randn(sampleNum, 100)#.to(device)

    print(sample_2[:10])
    
    sample_2 = Variable(sample_2).cuda()

    faceData =[]
    for model in model_list:
        output = model.decode(sample_2)


        #De-standardaize
        output_np = output.data.cpu().numpy()  #(batch, featureDim, frames)
        output_np = output_np*Xstd + Xmean

        output_np = np.swapaxes(output_np,1,2)  #(batch, frames, featureDim)
        output_np = np.reshape(output_np,(-1,featureDim))
        output_np = np.swapaxes(output_np,0,1)  #(featureDim, frames)

        #faceData = [output_np]
        faceData.append(output_np)
        
    glViewer.SetFaceParmData(faceData)
    glViewer.init_gl()



