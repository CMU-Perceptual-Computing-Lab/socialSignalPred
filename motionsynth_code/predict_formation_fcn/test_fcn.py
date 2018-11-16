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



test_data = np.load(datapath + test_dblist[0] + '.npz')

test_X_raw = test_data['clips']  #Input (1044,240,73)
test_Y_raw = test_data['speech']  #Input (1044,240,73)
#test_seqNames = test_data['seqNames']

#test_refPos_all = test_data['refPos'] #to go to the original position
#test_refRot_all = test_data['refRot'] #to go to the original orientation. Should take inverse for the quaternion


######################################
# Checkout Folder and pretrain file setting
checkpointRoot = './'

#No normalization version
checkpointFolder = checkpointRoot+ '/social_regressor_fcn_try6_noNorm_noReg/'
preTrainFileName= 'checkpoint_e150_loss33.4851.pth'






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
#features = (0,2,3,4,  5,7, 8,9)  #Ignoring Y axis
#else:
#features = (0,2, 3,5, 6,8, 9,11, 12,14, 15,17)  #Ignoring Y axis
#features = (0,2, 3,5, 6,8)  #Ignoring Y axis
#features = (0,2,3,5, 6,8)  #Pos + Face Rot

input_features = (0,2, 3,5,6,8,  9,11, 12,14, 15,17) #Position + FaceOri
#output_features = (0,2, 3,5) #Position + face normal
output_features = (0,2, 3,5, 6,8) #Position + face normal

## Test data
test_X = test_X_raw[0,:,:,:]      #(num, chunkLength, 9) //person0,1's all values (position, head orientation, body orientation)
test_X = np.concatenate( (test_X, test_X_raw[1,:,:,:]), axis= 2)      #(num, chunkLength, 18)
test_X = test_X[:,:,input_features]
test_Y = test_X_raw[2,:,:,:]    #2nd seller's position only
test_Y = test_Y[:,:,output_features]


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

    inputData_np = test_X_stdd[idxStart:(idxStart+batch_size),:,:]      
    inputData_np_ori = test_X[idxStart:(idxStart+batch_size),:,:]      #Just remember the last one

    #Reordering from (batchsize,featureDim,frames) ->(batch, frame,featureDim)
    #inputData_np = np.swapaxes(inputData, 1, 2) #(batch,  frame, featureDim)
    outputData_np = test_Y[idxStart:(idxStart+batch_size),:]   
    #outputData_np = outputData_np[:,:,np.newaxis]   #(batch, frame, 1)

    #numpy to Tensors
    inputData = Variable(torch.from_numpy(inputData_np)).cuda()
    outputGT = Variable(torch.from_numpy(outputData_np)).cuda()

    output = model(inputData)

    pred = output.data.cpu().numpy() #(batch, feature:6, frames)

    pred = pred*Ystd + Ymean   #(batch, feature:6, frames)

    pred = pred[0, :,:] #(feature:6, frames)
    inputData_np_ori = inputData_np_ori[0,:,:] #(feature:12, frames)
    vis_gt = outputData_np[0,:,:]  #(feature:6, frames)


    """Convert concatenated vector to parts"""
    pred_pos = pred[:2,:]
    pred_faceNorm = pred[2:4,:]
    pred_bodyNorm = pred[4:6,:]

    vis_gt_pos= vis_gt[:2,:]
    vis_gt_faceNorm= vis_gt[2:4,:]
    vis_gt_bodyNorm= vis_gt[4:6,:]


    vis_data_input_1_pos = inputData_np_ori[:2,:]  #0,1 for position
    vis_data_input_1_faceNorm = inputData_np_ori[2:4,:]  #0,1 for position
    vis_data_input_1_bodyNorm = inputData_np_ori[4:6,:]  #0,1 for position

    vis_data_input_2_pos = inputData_np_ori[6:8,:]  #0,1 for position
    vis_data_input_2_faceNorm = inputData_np_ori[8:10,:]  #0,1 for position
    vis_data_input_2_bodyNorm = inputData_np_ori[10:12,:]  #0,1 for position


    posData = [pred_pos, vis_gt_pos, vis_data_input_1_pos, vis_data_input_2_pos]

    # #Apply refTrans to go back to the original global position
    # for i in range(len(posData)): #data: (2,frames)
    #     frameLeng= posData[i].shape[1]

    #     data = posData[i]
    #     data_3d = np.zeros((data.shape[0]+1, data.shape[1])) #(3,frames)
    #     data_3d[0,:] = data[0,:]
    #     data_3d[2,:] = data[1,:]
    #     rotations = test_refRot[:frameLeng] #Quaternions. Take inverse
    #     data_3d = np.expand_dims(np.swapaxes(data_3d,0,1),1) #data_3d:(frames, jointNum,3)
    #     data_3d = rotations * data_3d  


    #     posData[i][0,:] = data_3d[:,0,0] + test_refPos[:frameLeng,0]
    #     posData[i][1,:] = data_3d[:,0,2] + test_refPos[:frameLeng,2]

    # posDataOri = posData
    glViewer.setPosOnly(posData)
    #glViewer.setPosOnly([vis_data_input_1_pos, vis_data_input_2_pos])
    #glViewer.SetFaceParmData([vis_data])

    faceNormalData = [pred_faceNorm, vis_gt_faceNorm, vis_data_input_1_faceNorm, vis_data_input_2_faceNorm]
    # for i in range(len(faceNormalData)): #data: (2,frames)
    #     frameLeng= faceNormalData[i].shape[1]

    #     data = faceNormalData[i]
    #     data_3d = np.zeros((data.shape[0]+1, data.shape[1])) #(3,frames)
    #     data_3d[0,:] = data[0,:]
    #     data_3d[2,:] = data[1,:]
    #     rotations = test_refRot[:frameLeng] #Quaternions. Take inverse

    #     data_3d = np.expand_dims(np.swapaxes(data_3d,0,1),1) #data_3d:(frames, jointNum,3)
    #     data_3d = rotations * data_3d  

    #     faceNormalData[i][0,:] = data_3d[:,0,0]
    #     faceNormalData[i][1,:] = data_3d[:,0,2]
    glViewer.setFaceNormal(faceNormalData)


    bodyNormalData = [pred_bodyNorm, vis_gt_bodyNorm, vis_data_input_1_bodyNorm, vis_data_input_2_bodyNorm]
    # for i in range(len(bodyNormalData)): #data: (2,frames)
    #     frameLeng= bodyNormalData[i].shape[1]

    #     data = bodyNormalData[i]
    #     data_3d = np.zeros((data.shape[0]+1, data.shape[1])) #(3,frames)
    #     data_3d[0,:] = data[0,:]
    #     data_3d[2,:] = data[1,:]
    #     rotations = test_refRot[:frameLeng] #Quaternions

    #     data_3d = np.expand_dims(np.swapaxes(data_3d,0,1),1) #data_3d:(frames, jointNum,3)
    #     data_3d = rotations * data_3d  

    #     bodyNormalData[i][0,:] = data_3d[:,0,0]
    #     bodyNormalData[i][1,:] = data_3d[:,0,2]

    # # glViewer.setFaceNormal([pred_faceNorm, vis_gt_faceNorm, vis_data_input_1_faceNorm, vis_data_input_2_faceNorm])
    glViewer.setBodyNormal(bodyNormalData)

    # glViewer.init_gl()

    """Generate Trajectory in Holden's form by pos and body orientation"""
    from Quaternions import Quaternions
    traj_list=[]
    initTrans_list=[]
    initRot_list=[]
    for i in range(len(posData)):
        targetPos = np.swapaxes(posData[i],0,1) #framesx2
        
        velocity = (targetPos[1:,:] - targetPos[:-1,:]).copy()      #(frames,2)
        frameLeng = velocity.shape[0]
        velocity_3d = np.zeros( (frameLeng,3) )
        velocity_3d[:,0] = velocity[:,0]
        velocity_3d[:,2] = velocity[:,1]        #(frames,3)
        velocity_3d = np.expand_dims(velocity_3d,1)   #(frames,1, 3)
        


        """ Compute Rvelocity"""
        targetNormal = np.swapaxes(bodyNormalData[i],0,1) #(frames, 2)
        forward = np.zeros( (targetNormal.shape[0],3) )
        forward[:,0]  = targetNormal[:,0]
        forward[:,2]  = targetNormal[:,1]
        forward = forward / np.sqrt((forward**2).sum(axis=-1))[...,np.newaxis]
        target = np.array([[0,0,1]]).repeat(len(forward), axis=0)
        rotation = Quaternions.between(forward, target)[:,np.newaxis]    #forward:(frame,3)

        velocity_3d = rotation[1:] * velocity_3d

        """ Get Root Rotation """
        rvelocity = Pivots.from_quaternions(rotation[1:] * -rotation[:-1]).ps


        """ Save output """
        trajData = np.zeros( (frameLeng,3))
        trajData[:,0] = velocity_3d[:,0,0] * 0.2    #0.2 to make holden data scale
        trajData[:,1] = velocity_3d[:,0,2] * 0.2    #0.2 to make holden data scale
        trajData[:,2] = rvelocity[:,0]


        initTrans = np.zeros( (1,3) )
        initTrans[0,0] = targetPos[0,0]
        initTrans[0,2] = targetPos[0,1]
        initTrans = initTrans*0.2

        trajData = np.swapaxes(trajData,0,1) 
        traj_list.append(trajData)
        initTrans_list.append(initTrans)


        initRot = -rotation[0] #Inverse to move [0,0,1] -> original Forward
        initRot_list.append(initRot)

        break

    glViewer.set_Holden_Trajectory_3(traj_list, initTrans=initTrans_list, initRot=initRot_list)
    glViewer.init_gl()

    # traj_list_seq.append(np.array(traj_list))

