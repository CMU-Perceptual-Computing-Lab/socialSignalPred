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

#test_dblist = ['data_hagglingSellers_speech_face_60frm_10gap_white_testing']
#test_dblist = ['data_hagglingSellers_speech_formation_30frm_10gap_white_testing']

#test_dblist = ['data_hagglingSellers_speech_formation_bySequence_white_testing']
#test_dblist = ['data_hagglingSellers_speech_formation_pNorm_bySequence_white_testing']

#test_dblist = ['data_hagglingSellers_speech_formation_pN_rN_rV_bySequence_white_testing_beta']


test_dblist = ['data_hagglingSellers_speech_formation_pN_rN_rV_bySequence_white_testing_4fcn']
test_dblist = ['data_hagglingSellers_speech_formation_bySequence_white_brl_testing_4fcn']   #no normalized

#test_dblist = ['data_hagglingSellers_speech_formation_pN_rotS_bySequence_white_training']


pkl_file = open(datapath + test_dblist[0] + '.pkl', 'rb')
test_data = pickle.load(pkl_file)
pkl_file.close()

test_X_raw_all = test_data['data']  #Input (1044,240,73)
test_Y_raw_all = test_data['speech']  #Input (1044,240,73)
test_seqNames = test_data['seqNames']


# test_refPos_all = test_data['refPos'] #to go to the original position
# test_refRot_all = test_data['refRot'] #to go to the original orientation. Should take inverse for the quaternion


######################################
# Checkout Folder and pretrain file setting
checkpointRoot = './'
checkpointFolder = checkpointRoot+ '/social_regressor_fcn/'
preTrainFileName= 'checkpoint_e150_loss0.0862.pth'

checkpointFolder = checkpointRoot+ '/social_regressor_fcn_try1/'
preTrainFileName= 'checkpoint_e4900_loss0.0867.pth'

checkpointFolder = checkpointRoot+ '/social_regressor_fcn_try19_noNorm/'
preTrainFileName= 'checkpoint_e102_loss19.7454.pth'
preTrainFileName= 'checkpoint_e300_loss22.0665.pth'

#Best: 24.594 No normalization version. with l1 regularization
checkpointFolder = checkpointRoot+ '/best/social_regressor_fcn_try6_noNorm_l1Reg/'
preTrainFileName= 'checkpoint_e45_loss17.9812.pth'


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

############################
## Import Traj2Body Model
import modelZoo_traj2Body
checkpointRoot = './'
checkpointFolder = checkpointRoot+ '/best/social_regressor_holden_73_try4/'
preTrainFileName= 'checkpoint_e5300_loss0.2356.pth'
preprocess_traj2body = np.load(checkpointFolder + 'preprocess_core.npz') #preprocess['Xmean' or 'Xstd']: (1, 73,1)

#model = getattr(modelZoo_traj2Body,args.model)().cuda()
model_traj2body = modelZoo_traj2Body.regressor_holden_73().cuda()
model_traj2body.eval()

#Create Model
trainResultName = checkpointFolder + preTrainFileName
loaded_state = torch.load(trainResultName, map_location=lambda storage, loc: storage)

model_traj2body.load_state_dict(loaded_state,strict=False) #strict False is needed some version issue for batchnorm
model_traj2body = model_traj2body.eval()  #Do I need this again?

tj2body_body_mean = preprocess_traj2body['body_mean']
tj2body_body_std = preprocess_traj2body['body_std']

tj2body_traj_mean = preprocess_traj2body['traj_mean']
tj2body_traj_std = preprocess_traj2body['traj_std']



############################
## Choose a sequence
#seqIdx =1

posErr_list = []
traj_list_seq =[]
bVisualize = True
for seqIdx in range(len(test_X_raw_all)):

    # if seqIdx!=len(test_X_raw_all)-1:
    #     continue

    for iteration in [1]:#[0,1]:  

        print('{}-{}'.format(os.path.basename(test_seqNames[seqIdx]), iteration))

        if iteration ==0:
            targetHumanIdx =1
            otherIdx =2
        else:
            targetHumanIdx =2
            otherIdx =1

        test_X_raw = test_X_raw_all[seqIdx]     #(3, frames, feature:9)
        test_Y_raw = test_Y_raw_all[seqIdx]     #(3, frames)


        # test_refPos = test_refPos_all[seqIdx]
        # test_refRot = - test_refRot_all[seqIdx] #Note the inverse


        #if test_X_raw.shape[-1]==5:
        #features = (0,2,3,4,  5,7, 8,9)  #Ignoring Y axis
        #else:
        #features = (0,2, 3,5, 6,8, 9,11, 12,14, 15,17)  #Ignoring Y axis
        features = (0,2, 3,5, 6,8)  #Ignoring Y axis
        #features = (0,2,3,5, 6,8)  #Pos + Face Rot
        test_X_raw = test_X_raw[:,:,features]

        test_X = test_X_raw[0,:,:]      #(num, 9) //person0,1's all values (position, head orientation, body orientation)
        test_X = np.concatenate( (test_X, test_X_raw[otherIdx,:,:]), axis= 1)      #(num, chunkLength, 18)
        #test_X = test_X[:,(0,1, 4,5)] #Pos only
        test_Y = test_X_raw[targetHumanIdx,:,:]    #2nd seller's position only

        test_X = np.swapaxes(np.expand_dims(test_X,0),1,2).astype(np.float32)  #(1, frames,feature:12)
        test_Y = np.swapaxes(np.expand_dims(test_Y,0),1,2).astype(np.float32)  #(1, frames,feature:6)

        #test_X_stdd = (test_X - preprocess['Xmean'][:,:,0]) / preprocess['Xstd'][:,:,0]     #(frames, feature:12)
        test_X_stdd = (test_X[:,:] - preprocess['Xmean']) / preprocess['Xstd']

        Ymean = preprocess['Ymean']
        Ystd = preprocess['Ystd']

        ######################################
        # Testing
        pred_all = np.empty([0,1],dtype=float)
        test_X_vis =np.empty([0,200],dtype=float)

        #idxStart  = bi*batch_size
        idxStart  = 0#bi*batch_size
        batch_size = test_X.shape[0]

        inputData_np = test_X_stdd#[idxStart:(idxStart+batch_size),:,:]      
        inputData_np_ori = test_X#[idxStart:(idxStart+batch_size),:,:]      #Just remember the last one
        
        #Reordering from (batchsize,featureDim,frames) ->(batch, frame,featureDim)
        #inputData_np = np.swapaxes(inputData, 1, 2) #(batch,  frame, featureDim)
        outputData_np = test_Y#[idxStart:(idxStart+batch_size),:]   
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
    

        # ####################################
        # ## Compute Errors
        # PosErr = (pred_pos - vis_gt_pos)**2           #pred_pos: (2,frames)
        # PosErr = np.sqrt(np.sum(PosErr, axis=0))
        # posErr_list.append(PosErr)

        posData = [pred_pos, vis_gt_pos, vis_data_input_1_pos, vis_data_input_2_pos]
        #Apply refTrans to go back to the original global position
        for i in range(len(posData)): #data: (2,frames)
            frameLeng= posData[i].shape[1]

            data = posData[i]
            data_3d = np.zeros((data.shape[0]+1, data.shape[1])) #(3,frames)
            data_3d[0,:] = data[0,:]
            data_3d[2,:] = data[1,:]
            #data_3d = utility.data_2dTo3D(data)
            #rotations = test_refRot[:frameLeng] #Quaternions. Take inverse
            data_3d = np.expand_dims(np.swapaxes(data_3d,0,1),1) #data_3d:(frames, jointNum,3)
            #data_3d = rotations * data_3d  

            posData[i][0,:] = data_3d[:,0,0]# + test_refPos[:frameLeng,0]
            posData[i][1,:] = data_3d[:,0,2]# + test_refPos[:frameLeng,2]

        # posDataOri = posData
        #glViewer.setPosOnly([vis_data_input_1_pos, vis_data_input_2_pos])
        #glViewer.SetFaceParmData([vis_data])

        faceNormalData = [pred_faceNorm, vis_gt_faceNorm, vis_data_input_1_faceNorm, vis_data_input_2_faceNorm]
        for i in range(len(faceNormalData)): #data: (2,frames)
            frameLeng= faceNormalData[i].shape[1]

            data = faceNormalData[i]
            data_3d = np.zeros((data.shape[0]+1, data.shape[1])) #(3,frames)
            data_3d[0,:] = data[0,:]
            data_3d[2,:] = data[1,:]
            #data_3d = utility.data_2dTo3D(data)
            #rotations = test_refRot[:frameLeng] #Quaternions. Take inverse

            data_3d = np.expand_dims(np.swapaxes(data_3d,0,1),1) #data_3d:(frames, jointNum,3)
            #data_3d = rotations * data_3d  

            faceNormalData[i][0,:] = data_3d[:,0,0]
            faceNormalData[i][1,:] = data_3d[:,0,2]


        bodyNormalData = [pred_bodyNorm, vis_gt_bodyNorm, vis_data_input_1_bodyNorm, vis_data_input_2_bodyNorm]
        for i in range(len(bodyNormalData)): #data: (2,frames)
            frameLeng= bodyNormalData[i].shape[1]

            data = bodyNormalData[i]
            data_3d = np.zeros((data.shape[0]+1, data.shape[1])) #(3,frames)
            data_3d[0,:] = data[0,:]
            data_3d[2,:] = data[1,:]
            #data_3d = utility.data_2dTo3D(data)
            #rotations = test_refRot[:frameLeng] #Quaternions

            data_3d = np.expand_dims(np.swapaxes(data_3d,0,1),1) #data_3d:(frames, jointNum,3)
            #data_3d = rotations * data_3d  

            bodyNormalData[i][0,:] = data_3d[:,0,0]
            bodyNormalData[i][1,:] = data_3d[:,0,2]

        # glViewer.setFaceNormal([pred_faceNorm, vis_gt_faceNorm, vis_data_input_1_faceNorm, vis_data_input_2_faceNorm])

        #glViewer.init_gl()

        """Generate Trajectory in Holden's form by pos and body orientation"""
        #traj_list, initTrans_list, initRot_list = utility.ConvertTrajectory_velocityForm(posData[:1], bodyNormalData[:1])
        traj_list, initTrans_list, initRot_list = utility.ConvertTrajectory_velocityForm(posData[1:2], bodyNormalData[1:2])      #GT version
        #glViewer.set_Holden_Trajectory_3(traj_list, initTrans=initTrans_list, initRot=initRot_list)
        # glViewer.init_gl()

        """ Apply Traj2Body """
        test_traj = traj_list[0] #(3, frames)
        test_traj = np.expand_dims(test_traj,0).astype(np.float32) #(num, 3, frameNum)

        ## Standardization
        test_traj_std = (test_traj - tj2body_traj_mean) / tj2body_traj_std

        inputData_np = test_traj_std
        inputData = Variable(torch.from_numpy(inputData_np)).cuda()  #(batch, 3, frameNum)
        output = model_traj2body(inputData)

        output_np = output.data.cpu().numpy()  #(batch, 73, frames)
        output_np = output_np[:,:69,:]      #crop the last 4, if there exists
        output_np = output_np*tj2body_body_std[:,:-4,:] + tj2body_body_mean[:,:-4,:]
        
        ## Optional: Overwrite global trans oreintation info
        #output_np[:,-3:,:] =  test_traj[:,:,:output_np.shape[2]]         

        output_np = np.swapaxes(output_np,1,2)  #(batch, frames, 73)
        output_np = np.reshape(output_np,(-1,69))
        output_np = np.swapaxes(output_np,0,1)

        if bVisualize==False:
            continue

        glViewer.setPosOnly(posData)
        glViewer.setFaceNormal(faceNormalData)
        glViewer.setBodyNormal(bodyNormalData)
        glViewer.set_Holden_Trajectory_3(traj_list, initTrans=initTrans_list, initRot=initRot_list)
        glViewer.set_Holden_Data_73([output_np],initTrans=initTrans_list,initRot=initRot_list)
        # traj_list_seq.append(np.array(traj_list))
        glViewer.init_gl()


# Compute error