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

import cPickle as pickle

# Utility Functions
import utility
from Quaternions import Quaternions
from utility import print_options,save_options
from utility import setCheckPointFolder
from utility import my_args_parser

#by jhugestar
sys.path.append('/ssd/codes/glvis_python/')
#from glViewer import SetFaceParmData,setSpeech,setSpeechGT,setSpeech_binary, setSpeechGT_binary, init_gl #opengl visualization 
import glViewer

sys.path.append('/ssd/codes/pytorch_motionSynth/motionsynth_data/motion')
from Pivots import Pivots


######################################
# Logging
import logging
#FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
FORMAT = '[%(levelname)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)  ##default logger


######################################
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

#test_dblist = ['data_hagglingSellers_speech_group_face_120frm_5gap_white_5dim_training_tiny']
test_dblist = ['data_hagglingSellers_speech_group_face_5dim_bySequence_testing']
test_dblist = ['data_hagglingSellers_speech_group_face_5dim_bySequence_training_tiny']

#train_data = np.load(datapath + train_dblist[0] + '.npz')
pkl_file = open(datapath + test_dblist[0] + '.pkl', 'rb')
test_data = pickle.load(pkl_file)
pkl_file.close()

test_X_raw_all = test_data['data']  #Input (1044,240,73)
# test_Y_raw_all = test_data['speech']  #Input (1044,240,73)
test_seqNames = test_data['seqNames']

######################################
# Checkout Folder and pretrain file setting


checkpointRoot = './'
checkpointFolder = checkpointRoot+ '/social_regressor_fcn_bn_encoder_try5/'
preTrainFileName= 'checkpoint_e200_loss0.6707.pth'


checkpointFolder = checkpointRoot+ '/social_regressor_fcn_bn_encoder_try7/'
preTrainFileName= 'checkpoint_e450_loss0.6481.pth'



#120 frames good!

bVisualize = True

######################################
# Load Data pre-processing
preprocess = np.load(checkpointFolder + 'preprocess_core.npz') #preprocess['Xmean' or 'Xstd']: (1, 73,1)


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
batch_size = 500#512
criterion = nn.BCELoss()

#Creat Model
#model = modelZoo.naive_mlp_wNorm_2().cuda()
model = getattr(modelZoo,args.model)().cuda()

model.eval()

#Creat Model
trainResultName = checkpointFolder + preTrainFileName
loaded_state = torch.load(trainResultName, map_location=lambda storage, loc: storage)

model.load_state_dict(loaded_state,strict=False) #strict False is needed some version issue for batchnorm
model = model.eval()  #Do I need this again?




############################################################################
# Load Pretrained Autoencoder

######################################
# Checkout Folder and pretrain file setting
ae_checkpointRoot = './'
ae_checkpointFolder = checkpointRoot+ '/social_autoencoder_first_try4/'
preTrainFileName= 'checkpoint_e100_loss0.0351.pth'

# ######################################
# # Load Pretrained Auto-encoder
ae_preprocess = np.load(ae_checkpointFolder + 'preprocess_core.npz') #preprocess['Xmean' or 'Xstd']: (1, 73,1))
ae_model = modelZoo.autoencoder_first(5).cuda()

#Creat Model
trainResultName = ae_checkpointFolder + preTrainFileName
loaded_state = torch.load(trainResultName, map_location=lambda storage, loc: storage)

ae_model.load_state_dict(loaded_state,strict=False) #strict False is needed some version issue for batchnorm
ae_model = ae_model.eval()  #Do I need this again?


######################################
# If model is for single frame data
bSingleFrameInput = False
if not hasattr(model, 'init_hidden'):
    bSingleFrameInput = True

############################
## Choose a sequence
#seqIdx =1

posErr_list = []
skeletonErr_list = []


for seqIdx in range(len(test_X_raw_all)):

    # if seqIdx!=len(test_X_raw_all)-1:
    #     continue

    print('{}'.format(os.path.basename(test_seqNames[seqIdx])))

    test_X_raw = test_X_raw_all[seqIdx]     #(1, frames, feature:73)
    ######################################
    # Input/Output Option

    ## Test data
    test_X = test_X_raw[0:1,:,:]      #(1, frames, features:73) //person0,1's all values (position, head orientation, body orientation)
    test_X = np.concatenate( (test_X, test_X_raw[1:2,:,:]), axis= 2)      #(1, frames, features:146)
    test_Y = test_X_raw[2:3,:,:]    #(1, frames, features:73)

    ######################################
    # Data pre-processing
    test_X = np.swapaxes(test_X, 1, 2).astype(np.float32)   #(1, features, frames)
    test_Y = np.swapaxes(test_Y, 1, 2).astype(np.float32)   #(1, features, frames)


    ######################################
    # Data pre-processing
    preprocess = np.load(checkpointFolder + 'preprocess_core.npz') #preprocess['Xmean' or 'Xstd']: (1, 73,1)

    body_mean = preprocess['body_mean']
    body_std = preprocess['body_std']

    body_mean_two = preprocess['body_mean_two']
    body_std_two = preprocess['body_std_two']

    test_X_std = (test_X - body_mean_two) / body_std_two
    test_Y_std = (test_Y - body_mean) / body_std


    idxStart  = 0
    inputData_np = test_X_std[idxStart:(idxStart+batch_size),:,:]  #(batch, 3, frameNum)
    outputData_np = test_Y_std[idxStart:(idxStart+batch_size),:,:] #(batch, 73 , frameNum)
    
    inputData_np_ori = test_X[idxStart:(idxStart+batch_size),:,:]  #(batch, 3, frameNum)
    outputData_np_GT = test_Y[idxStart:(idxStart+batch_size),:,:] #(batch, 73, frameNum)

    inputData = Variable(torch.from_numpy(inputData_np)).cuda()  #(batch, 3, frameNum)
    outputGT = Variable(torch.from_numpy(outputData_np)).cuda()  #(batch, 73, frameNum)

    # ===================forward=====================
    output = model(inputData)
    output = ae_model.decoder(output)
    #loss = criterion(output, outputGT)
    #loss = criterion(output, outputGT)


    #De-standardaize
    output_np = output.data.cpu().numpy()  #(batch, featureDim:5, frames)
    output_np = output_np*body_std + body_mean

    output_np = np.squeeze(output_np)  #(featureDim:5, 73)

    """Overwrite global trans oreintation info"""
    #output_np[-7:-3,:] =  outputData_np_GT[0,-7:-3,:output_np.shape[1]]         #


    #Output GT
    outputData_np_GT = np.squeeze(outputData_np_GT)  #(batch, frames, 73)
  
    #Input GTs
    inputData_np_ori_1 = inputData_np_ori[:,:5,:]
    inputData_np_ori_1 =  np.squeeze(inputData_np_ori_1)  #(batch, frames, 73)

    inputData_np_ori_2 = inputData_np_ori[:,5:,:]
    inputData_np_ori_2 = np.squeeze(inputData_np_ori_2)  #(batch, frames, 73)

    #glViewer.show_Holden_Data_73([ outputData_np_ori, inputData_np_ori, output_np] )

    frameLen = output_np.shape[1]
    faceData = [ output_np, outputData_np_GT[:,:frameLen], inputData_np_ori_1[:,:frameLen], inputData_np_ori_2[:,:frameLen] ]


    """Remvoe OUTPU GT"""
    #faceData = [faceData[0],faceData[2],faceData[3]]
    faceData = [faceData[0],faceData[1]]


    # # ####################################
    # # ## Compute Skeleton Error
    # HOLDEN_DATA_SCALING = 5
    # bodyData_pred = bodyData[0][:-7,:]*HOLDEN_DATA_SCALING   #prediction (66,frames)
    # """Baselines"""
    # """
    # #bodyData_pred = bodyData[2][:-7,:]*HOLDEN_DATA_SCALING   #Baseline:Mirroring (buyer)
    # #bodyData_pred = bodyData[3][:-7,:]*HOLDEN_DATA_SCALING   #Baseline: Mirroring (other seller)
    # bodyData_pred = body_mean.copy()[0,:66,:]   #Mirroring (buyer)   (73)
    # bodyData_pred = np.repeat(bodyData_pred,bodyData[0].shape[1],axis=1)*HOLDEN_DATA_SCALING
    # """

    # bodyData_gt = bodyData[1][:-7,:]*HOLDEN_DATA_SCALING       #GT
    # bodyData_gt = bodyData_gt[:,:bodyData_pred.shape[1]]
    # skelErr = ( bodyData_pred -  bodyData_gt)**2           #66, frames
    # skelErr = np.reshape(skelErr, (3,22,-1))        #3,22, frames
    # skelErr = np.sqrt(np.sum(skelErr, axis=0))      #22,frames        
    # skelErr = np.mean(skelErr,axis=0)   #frames
    # skeletonErr_list.append(skelErr)
    
    if bVisualize==False:
        continue

    glViewer.SetFaceParmData(faceData,False)
    glViewer.init_gl()

# Compute error

##Draw Error Figure
avg_skelErr_list=[]
total_avg_skelErr = 0
cnt=0
for p in skeletonErr_list:
    avgValue = np.mean(p)
    avg_skelErr_list.append(avgValue)
    print(avgValue)
    total_avg_skelErr += avgValue*len(p)
    cnt += len(p)

total_avg_skelErr = total_avg_skelErr/cnt
std = np.std(avg_skelErr_list)
print("total_avg_skelErr: {}, std {}".format(total_avg_skelErr, std))


# add a subplot with no frame
bShowGraph = True
if bShowGraph:
    import matplotlib.pyplot as plt
    plt.rc('xtick', labelsize=18)     
    plt.rc('ytick', labelsize=18)
    

    ax2=plt.subplot(111)
    plt.plot(avg_skelErr_list)
    plt.title('Average Pos Error', fontsize=20)
    plt.grid()
    plt.xlabel('Seq. Index', fontsize=20)
    plt.ylabel('Error (cm)', fontsize=20)
    
    plt.tight_layout()
    plt.show()




