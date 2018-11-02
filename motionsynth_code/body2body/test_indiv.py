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

import cPickle as pickle
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

rng = np.random.RandomState()
# rng = np.random.RandomState(23456)
# torch.manual_seed(23456)
# torch.cuda.manual_seed(23456)


######################################
# Dataset 
#datapath ='/ssd/codes/pytorch_motionSynth/motionsynth_data' 
datapath ='../../motionsynth_data/data/processed/' 

#test_dblist = ['data_hagglingSellers_speech_face_60frm_10gap_white_testing']
test_dblist = ['data_hagglingSellers_speech_body_group_240frm_15gap_white_noGa_testing_tiny']

pkl_file = open(datapath + test_dblist[0] + '.pkl', 'rb')
test_data = pickle.load(pkl_file)
pkl_file.close()
test_X_raw= test_data['data']      #Input (numClip, frames, featureDim:73)

######################################
# Checkout Folder and pretrain file setting
checkpointRoot = './'
checkpointFolder = checkpointRoot+ '/social_regressor_fcn_try5/'
preTrainFileName= 'checkpoint_e1074_loss0.1115.pth'


######################################
# Input/Output Option

## Test data
test_X = test_X_raw[0,:,:,:]      #(num, chunkLength, 9) //person0,1's all values (position, head orientation, body orientation)
test_X = np.concatenate( (test_X, test_X_raw[1,:,:,:]), axis= 2)      #(num, chunkLength, 18)
test_Y = test_X_raw[2,:,:,:]    #2nd seller's position only

test_X_swap = test_X_raw[0,:,:,:]      #(num, chunkLength, 9) //person0,1's all values (position, head orientation, body orientation)
test_X_swap = np.concatenate( (test_X_swap, test_X_raw[2,:,:,:]), axis= 2)      #(num, chunkLength, 18)
test_Y_swap = test_X_raw[1,:,:,:]    #2nd seller's position only

test_X = np.concatenate( (test_X, test_X_swap), axis=0)
test_Y = np.concatenate( (test_Y, test_Y_swap), axis=0)

######################################
# Data pre-processing

test_X = np.swapaxes(test_X, 1, 2).astype(np.float32) 
test_Y = np.swapaxes(test_Y, 1, 2).astype(np.float32) 


######################################
# Data pre-processing
preprocess = np.load(checkpointFolder + 'preprocess_core.npz') #preprocess['Xmean' or 'Xstd']: (1, 73,1)

body_mean = preprocess['body_mean']
body_std = preprocess['body_std']

body_mean_two = preprocess['body_mean_two']
body_std_two = preprocess['body_std_two']

test_X_std = (test_X - body_mean_two) / body_std_two
test_Y_std = (test_Y - body_mean) / body_std

######################################
# Load Options
log_file_name = os.path.join(checkpointFolder, 'opt.json')
import json
with open(log_file_name, 'r') as opt_file:
    options_dict = json.load(opt_file)

    ##Setting Options to Args
    args.model = options_dict['model']


######################################
# Network Setting
#batch_size = args.batch
batch_size = 1#512
#criterion = nn.BCELoss()
criterion = nn.MSELoss()

#Creat Model
#model = modelZoo.autoencoder_first().cuda()
# featureDim = test_X_raw.shape[2]
# latentDim = 200
# model = modelZoo.autoencoder_3conv_vect_vae(featureDim,latentDim).cuda()
model = getattr(modelZoo,args.model)().cuda()
model.eval()

#Creat Model
trainResultName = checkpointFolder + preTrainFileName
loaded_state = torch.load(trainResultName, map_location=lambda storage, loc: storage)

model.load_state_dict(loaded_state,strict=False) #strict False is needed some version issue for batchnorm
model = model.eval()  #Do I need this again?


# Ystd = np.swapaxes(Ystd,1,2)
# Ymean = np.swapaxes(Ymean,1,2)

######################################
# Training
batchinds = np.arange(test_X.shape[0] // batch_size)
pred_all = np.empty([0,1],dtype=float)
test_X_vis =np.empty([0,200],dtype=float)

featureDim = 10
for _, bi in enumerate(batchinds):

    if bi %10 !=0:
        continue

    idxStart  = bi*batch_size 
    inputData_np = test_X_std[idxStart:(idxStart+batch_size),:,:]  #(batch, 3, frameNum)
    outputData_np = test_Y_std[idxStart:(idxStart+batch_size),:,:] #(batch, 73 , frameNum)
    
    inputData_np_ori = test_X[idxStart:(idxStart+batch_size),:,:]  #(batch, 3, frameNum)
    outputData_np_ori = test_Y[idxStart:(idxStart+batch_size),:,:] #(batch, 73, frameNum)

    inputData = Variable(torch.from_numpy(inputData_np)).cuda()  #(batch, 3, frameNum)
    outputGT = Variable(torch.from_numpy(outputData_np)).cuda()  #(batch, 73, frameNum)

    # ===================forward=====================
    output = model(inputData)
    #loss = criterion(output, outputGT)
    #loss = criterion(output, outputGT)


    #De-standardaize
    output_np = output.data.cpu().numpy()  #(batch, 73, frames)
    output_np = output_np*body_std + body_mean

    output_np = np.swapaxes(output_np,1,2)  #(batch, frames, 73)
    output_np = np.reshape(output_np,(-1,73))
    output_np = np.swapaxes(output_np,0,1)


    #Output GT
    outputData_np_ori = np.swapaxes(outputData_np_ori,1,2)  #(batch, frames, 73)
    outputData_np_ori = np.reshape(outputData_np_ori,(-1,73))
    outputData_np_ori = np.swapaxes(outputData_np_ori,0,1)


    #Input GTs
    inputData_np_ori_1 = inputData_np_ori[:,:73,:]
    inputData_np_ori_1 = np.swapaxes(inputData_np_ori_1,1,2)  #(batch, frames, 73)
    inputData_np_ori_1 = np.reshape(inputData_np_ori_1,(-1,73))
    inputData_np_ori_1 = np.swapaxes(inputData_np_ori_1,0,1)

    inputData_np_ori_2 = inputData_np_ori[:,73:,:]
    inputData_np_ori_2 = np.swapaxes(inputData_np_ori_2,1,2)  #(batch, frames, 73)
    inputData_np_ori_2 = np.reshape(inputData_np_ori_2,(-1,73))
    inputData_np_ori_2 = np.swapaxes(inputData_np_ori_2,0,1)


    
    #glViewer.show_Holden_Data_73([ outputData_np_ori, inputData_np_ori, output_np] )

    glViewer.set_Holden_Data_73([output_np, outputData_np_ori, inputData_np_ori_1, inputData_np_ori_2] )
    #glViewer.set_Holden_Trajectory_3([inputData_np_ori, inputData_np_ori] )
    glViewer.init_gl()

