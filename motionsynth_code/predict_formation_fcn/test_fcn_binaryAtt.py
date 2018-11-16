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


#BRL, normalization by the first frame
test_dblist = ['data_hagglingSellers_speech_formation_240frm_5gap_white_brl_firstFrmNorm_testing_tiny']  

#BRL, no normalization
test_dblist = ['data_hagglingSellers_speech_formation_240frm_5gap_white_brl_testing']       
test_dblist = ['data_hagglingSellers_speech_formation_240frm_5gap_white_brl_testing_tiny']
test_dblist = ['data_hagglingSellers_speech_formation_240frm_5gap_white_brl_training']


test_dblist = ['data_hagglingSellers_speech_formation_240frm_5gap_white_brl_atten_testing_tiny']
test_dblist = ['data_hagglingSellers_speech_formation_240frm_5gap_white_brl_atten_training']



test_data = np.load(datapath + test_dblist[0] + '.npz')

test_X_raw = test_data['clips']  #Input (1044,240,73)
test_Y_raw = test_data['speech']  #Input (1044,240,73)
#test_seqNames = test_data['seqNames']

test_attention_raw = test_data['attention'] #[pIdx](chunkNum, frames, 73)

#test_refPos_all = test_data['refPos'] #to go to the original position
#test_refRot_all = test_data['refRot'] #to go to the original orientation. Should take inverse for the quaternion


######################################
# Checkout Folder and pretrain file setting
checkpointRoot = './'

#No normalization version
checkpointFolder = checkpointRoot+ '/best/social_regressor_fcn_try6_noNorm_noReg/'
preTrainFileName= 'checkpoint_e150_loss33.4851.pth'

checkpointFolder = checkpointRoot+ 'social_regressor_fcn_att_try8/'
preTrainFileName= 'checkpoint_e976_loss2.1299.pth'

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


otherIdx =1
targetHumanIdx =2

#if test_X_raw.shape[-1]==5:
input_features = (0,2,  9,11) #Position only
output_features = (0,2) #Position only + face normal

## Test data
test_X = test_X_raw[0,:,:,:]      #(num, chunkLength, 9) //person0,1's all values (position, head orientation, body orientation)
test_X = np.concatenate( (test_X, test_X_raw[1,:,:,:]), axis= 2)      #(num, chunkLength, 18)
test_X = test_X[:,:,input_features]
test_Y = test_X_raw[2,:,:,:]    #2nd seller's position only
test_Y = test_Y[:,:,output_features]

##Concatenated Attentions
test_X_attention = test_attention_raw[0]    #(chunkNum, frames, feagureDim:2)
test_X_attention = np.concatenate( (test_X_attention, test_attention_raw[1]), axis= 2)      #(chunkNum, frames, featureDim:4)
test_Y_attention = test_attention_raw[2]  #(chunkNum, frames, featureDim:2)


test_X = np.concatenate( (test_X, test_X_attention.copy()), axis= 2 )     #(num, chunkLength,4+4 )
test_Y = np.concatenate( (test_Y, test_Y_attention.copy()), axis= 2 )      #(num, chunkLength,2+2 )

test_X_attention = np.swapaxes(test_X_attention, 1, 2).astype(np.float32)   #(chunkNum, featureDim:2, frames)
test_Y_attention = np.swapaxes(test_Y_attention, 1, 2).astype(np.float32) #(chunkNum, featureDim:2, frames)

test_X = np.swapaxes(test_X, 1, 2).astype(np.float32) 
test_Y = np.swapaxes(test_Y, 1, 2).astype(np.float32) 

#test_X_stdd = (test_X - preprocess['Xmean'][:,:,0]) / preprocess['Xstd'][:,:,0]     #(frames, feature:12)
test_X_stdd = (test_X[:,:] - preprocess['Xmean']) / preprocess['Xstd']

Ymean = preprocess['Ymean']
Ystd = preprocess['Ystd']

######################################
# Testing
pred_all = np.empty([0,1],dtype=float)
test_X_vis =np.empty([0,200],dtype=float)
batch_size=1


batchinds = np.arange(test_X.shape[0] // batch_size)
rng.shuffle(batchinds)

for bii, bi in enumerate(batchinds):

    #idxStart  = bi*batch_size
    idxStart  = bi*batch_size
    #batch_size = test_X.shape[0]

    inputData_np = test_X_stdd[idxStart:(idxStart+batch_size),:,:]      #(batch, featureDim:8, frames)
    inputData_np_ori = test_X[idxStart:(idxStart+batch_size),:,:]       #(batch, featureDim:8, frames)
    outputData_np = test_Y[idxStart:(idxStart+batch_size),:]            #(batch, featureDim:4, frames)

    #inputData_np_attention = test_X_attention[idxStart:(idxStart+batch_size),:,:]   #(featureDim:4, frames)
    #outputData_np_attention = test_Y_attention[idxStart:(idxStart+batch_size),:,:]   #(featureDim: 2, frames)


    #numpy to Tensors
    inputData = Variable(torch.from_numpy(inputData_np)).cuda()
    outputGT = Variable(torch.from_numpy(outputData_np)).cuda()

    output = model(inputData)

    pred = output.data.cpu().numpy() #(batch, feature:6, frames)

    pred = pred*Ystd + Ymean   #(batch, feature:6, frames)

    pred = pred[0, :,:] #(feature:6, frames)
    inputData_np_ori = inputData_np_ori[0,:,:] #(feature:4, frames)
    vis_gt = outputData_np[0,:,:]  #(feature:2, frames)

    """Convert concatenated vector to parts"""
    pred_pos = pred[:2,:]
    # pred_faceNorm = pred[2:4,:]
    # pred_bodyNorm = pred[4:6,:]
    pred_faceNorm = pred[2,:]
    pred_bodyNorm = pred[3,:]

    

    vis_gt_pos= vis_gt[:2,:]
    vis_gt_faceNorm= vis_gt[2,:]
    vis_gt_bodyNorm= vis_gt[3,:]

    vis_data_input_1_pos = inputData_np_ori[:2,:]  #0,1 for position
    vis_data_input_1_faceNorm = inputData_np_ori[4,:]  #0,1 for position
    vis_data_input_1_bodyNorm = inputData_np_ori[5,:]  #0,1 for position

    vis_data_input_2_pos = inputData_np_ori[2:4,:]  #0,1 for position
    vis_data_input_2_faceNorm = inputData_np_ori[6,:]  #0,1 for position
    vis_data_input_2_bodyNorm = inputData_np_ori[7,:]  #0,1 for position

    posData = [pred_pos, vis_gt_pos, vis_data_input_1_pos, vis_data_input_2_pos]
    faceNormalData = [pred_faceNorm, vis_gt_faceNorm, vis_data_input_1_faceNorm, vis_data_input_2_faceNorm]
    bodyNormalData = [pred_bodyNorm, vis_gt_bodyNorm, vis_data_input_1_bodyNorm, vis_data_input_2_bodyNorm]


    """ Attention """
    # gt_attention_face = outputData_np_attention[0,0,:]  #(frame,)
    # gt_attention_body = outputData_np_attention[0,1,:] #(frame,)

    # attention_face_b = inputData_np_attention[0,0,:] #(frame,)
    # attention_body_b = inputData_np_attention[0,1,:] #(frame,)

    # attention_face_r = inputData_np_attention[0,2,:] #(frame,)
    # attention_body_r = inputData_np_attention[0,3,:] #(frame,)


    import attention
    #Debug: compute face normal from attention and put that as body normal
    #leftPos = posData[1]    #target GT
    leftPos = posData[0]    #prediction
    buyerPos = posData[2]
    rightPos = posData[3]
    faceNormalData[0] = attention.attention2Direction(leftPos, buyerPos, rightPos, pred_faceNorm)
    bodyNormalData[0] = attention.attention2Direction(leftPos, buyerPos, rightPos, pred_bodyNorm)


    leftPos = posData[1]    #GT

    """Convert attention to direction"""
    faceNormalData[1] = attention.attention2Direction(leftPos, buyerPos, rightPos, faceNormalData[1])
    faceNormalData[2] = attention.attention2Direction(buyerPos,rightPos, leftPos, faceNormalData[2])
    faceNormalData[3] = attention.attention2Direction(rightPos, leftPos, buyerPos, faceNormalData[3])

    bodyNormalData[1] = attention.attention2Direction(leftPos, buyerPos, rightPos, bodyNormalData[1])
    bodyNormalData[2] = attention.attention2Direction(buyerPos,rightPos, leftPos, bodyNormalData[2])
    bodyNormalData[3] = attention.attention2Direction(rightPos, leftPos, buyerPos, bodyNormalData[3])

    glViewer.setPosOnly(posData)
    glViewer.setFaceNormal(faceNormalData)
    glViewer.setBodyNormal(bodyNormalData)

    """Generate Trajectory in Holden's form by pos and body orientation"""
    """
    from utility import ConvertTrajectory_velocityForm
    traj_list, initTrans_list,initRot_list = ConvertTrajectory_velocityForm([posData[0]],[bodyNormalData[0]])
    glViewer.set_Holden_Trajectory_3(traj_list, initTrans=initTrans_list, initRot=initRot_list)
    """

    glViewer.init_gl()

    # traj_list_seq.append(np.array(traj_list))



# import matplotlib.pyplot as plt
#     ax2=plt.subplot(311)
#     plt.plot(avg_posErr_list)
#     plt.title('Average Pos Error')