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

from utility import data_2dTo3D
from utility import ConvertTrajectory_velocityForm

from sklearn.preprocessing import normalize

sys.path.append('../../motionsynth_data/motion')
#import BVH as BVH
#import Animation as Animation
from Quaternions import Quaternions
from Pivots import Pivots





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
#test_dblist = ['data_hagglingSellers_speech_formation_pN_rN_rV_bySequence_white_testing_4fcn']


test_dblist = ['data_hagglingSellers_speech_formation_pN_rN_rV_bySequence_white_brl_testing_4fcn']  #normalized
test_dblist = ['data_hagglingSellers_speech_formation_bySequence_white_brl_firstFrmNorm_testing_4fcn']   #normalized by the first frame
test_dblist = ['data_hagglingSellers_speech_formation_bySequence_white_brl_testing_4fcn']   #no normalized
test_dblist = ['data_hagglingSellers_speech_formation_pN_rN_rV_bySequence_white_brl_training_4fcn_atten']   #no normalized
test_dblist = ['data_hagglingSellers_speech_formation_pN_rN_rV_bySequence_white_brl_testing_4fcn_atten']   #no normalized

test_dblist = ['data_hagglingSellers_speech_formation_bySequence_white_brl_training_4fcn_atten']   #no normalized, binary attention for orientation
test_dblist = ['data_hagglingSellers_speech_formation_bySequence_white_brl_testing_4fcn_atten']   #no normalized, binary attention for orientation




#test_dblist = ['data_hagglingSellers_speech_formation_pN_rotS_bySequence_white_training']


pkl_file = open(datapath + test_dblist[0] + '.pkl', 'rb')
test_data = pickle.load(pkl_file)
pkl_file.close()

test_X_raw_all = test_data['data']  #Input (1044,240,73)
test_Y_raw_all = test_data['speech']  #Input (1044,240,73)
test_seqNames = test_data['seqNames']

test_attention_all = test_data['attention'] #face,body 

# test_refPos_all = test_data['refPos'] #to go to the original position
# test_refRot_all = test_data['refRot'] #to go to the original orientation. Should take inverse for the quaternion


######################################
# Checkout Folder and pretrain file setting
checkpointRoot = './'

#Best: 24.594 No normalization version. with l1 regularization
checkpointFolder = checkpointRoot+ '/best/social_regressor_fcn_try6_noNorm_l1Reg/'
preTrainFileName= 'checkpoint_e45_loss17.9812.pth'

checkpointFolder = checkpointRoot+ '/social_regressor_fcn_att_try23/'
preTrainFileName= 'checkpoint_e350_loss26.7967.pth'
preTrainFileName= 'checkpoint_e14200_loss26.0026.pth'




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
## Choose a sequence
#seqIdx =1

posErr_list = []
faceOriErr_list = []
bodyOriErr_list = []
traj_list_seq =[]
bVisualize = False
for seqIdx in range(len(test_X_raw_all)):

    # if seqIdx!=len(test_X_raw_all)-1:
    #     continue

    for iteration in [1]:#[0,1]:  

        seqName_base = os.path.basename(test_seqNames[seqIdx])
        # if bVisualize == True and not ('170228_haggling_b2_group1' in seqName_base):
        #     continue

        print('{}-{}'.format(seqName_base, iteration))

        if iteration ==0:
            targetHumanIdx =1
            otherIdx =2
        else:
            targetHumanIdx =2
            otherIdx =1

        test_X_raw = test_X_raw_all[seqIdx]     #(3, frames, feature:9)
        test_Y_raw = test_Y_raw_all[seqIdx]     #(3, frames)

        test_attention_raw = test_attention_all[seqIdx]

   
        features = (0,2)  #Ignoring Y axis
        test_X_raw = test_X_raw[:,:,features]

        test_X = test_X_raw[0,:,:]      #(num, 9) //person0,1's all values (position, head orientation, body orientation)
        test_X = np.concatenate( (test_X, test_X_raw[otherIdx,:,:]), axis= 1)      #(num, chunkLength, 18)

        test_Y = test_X_raw[targetHumanIdx,:,:]    #2nd seller's position only

        
        ##Concatenated Attentions
        test_X_attention = test_attention_raw[0]    #(chunkNum, frames, feagureDim:2)
        test_X_attention = np.concatenate( (test_X_attention, test_attention_raw[otherIdx]), axis= 1)      #(frames, featureDim:4)
        test_Y_attention = test_attention_raw[targetHumanIdx]  #(chunkNum, frames, featureDim:2)

        test_X = np.concatenate( (test_X, test_X_attention.copy()), axis= 1 )     #(frames,4+4 )
        test_Y = np.concatenate( (test_Y, test_Y_attention.copy()), axis= 1 )      #(frames,2+2 )

        test_X = np.swapaxes(np.expand_dims(test_X,0),1,2).astype(np.float32)  #(1, frames,feature:8)
        test_Y = np.swapaxes(np.expand_dims(test_Y,0),1,2).astype(np.float32)  #(1, frames,feature:4)

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
        pred_faceNorm = pred[2,:]
        pred_bodyNorm = pred[3,:]

        vis_gt_pos= vis_gt[:2,:]
        vis_gt_faceNorm= vis_gt[2,:]
        vis_gt_bodyNorm= vis_gt[3,:]

        #gt_attention_face = test_attention[targetHumanIdx,:,0]  #(frame,)
        #gt_attention_body = test_attention[targetHumanIdx,:,1] #(frame,)

        vis_data_input_b_pos = inputData_np_ori[:2,:]  #0,1 for position
        vis_data_input_b_faceNorm = inputData_np_ori[4,:]  #0,1 for position
        vis_data_input_b_bodyNorm = inputData_np_ori[5,:]  #0,1 for position

        #attention_face_b = test_attention[0,:,0] #(frame,)
        #attention_body_b = test_attention[0,:,1] #(frame,)
        
        # vis_data_input_1_faceAttention = test_attention[0,:,0]
        # vis_data_input_1_bodyAttention = test_attention[0,:,1]

        vis_data_input_r_pos = inputData_np_ori[2:4,:]  #0,1 for position
        vis_data_input_r_faceNorm = inputData_np_ori[6,:]  #0,1 for position
        vis_data_input_r_bodyNorm = inputData_np_ori[7,:]  #0,1 for position

        #attention_face_r = test_attention[otherIdx,:,0] #(frame,)
        #attention_body_r = test_attention[otherIdx,:,1] #(frame,)

        # vis_data_input_2_faceAttention = test_attention[0,:,0]
        # vis_data_input_2_bodyAttention = test_attention[0,:,1]
    
        posData = [pred_pos, vis_gt_pos, vis_data_input_b_pos, vis_data_input_r_pos]
        faceNormalData = [pred_faceNorm, vis_gt_faceNorm, vis_data_input_b_faceNorm, vis_data_input_r_faceNorm]
        bodyNormalData = [pred_bodyNorm, vis_gt_bodyNorm, vis_data_input_b_bodyNorm, vis_data_input_r_bodyNorm]

        #Make sure the frame lengths are the same
        frameLen = pred_pos.shape[1]
        posData = [ p[:,:frameLen] for p in posData]
        faceNormalData = [ p[:frameLen] for p in faceNormalData]
        bodyNormalData = [ p[:frameLen] for p in bodyNormalData]
        #gt_attention_face = gt_attention_face[:frameLen]
        #gt_attention_body = gt_attention_body[:frameLen]

        """Convert attention to direction"""
        import attention
        #Debug: compute face normal from attention and put that as body normal
        leftPos = posData[0]    #target GT
        buyerPos = posData[2]
        rightPos = posData[3]
        faceNormalData[0] = attention.attention2Direction(leftPos, buyerPos, rightPos, pred_faceNorm)
        bodyNormalData[0] = attention.attention2Direction(leftPos, buyerPos, rightPos, pred_bodyNorm)

        leftPos = posData[1]    #GT

        faceNormalData[1] = attention.attention2Direction(leftPos, buyerPos, rightPos, faceNormalData[1])
        faceNormalData[2] = attention.attention2Direction(buyerPos,rightPos, leftPos, faceNormalData[2])
        faceNormalData[3] = attention.attention2Direction(rightPos, leftPos, buyerPos, faceNormalData[3])

        bodyNormalData[1] = attention.attention2Direction(leftPos, buyerPos, rightPos, bodyNormalData[1])
        bodyNormalData[2] = attention.attention2Direction(buyerPos,rightPos, leftPos, bodyNormalData[2])
        bodyNormalData[3] = attention.attention2Direction(rightPos, leftPos, buyerPos, bodyNormalData[3])
        

        # ####################################
        # ## Compute Errors
        PosErr = (pred_pos - vis_gt_pos[:,:pred_pos.shape[1]])**2           #pred_pos: (2,frames)
        PosErr = np.sqrt(np.sum(PosErr, axis=0))
        posErr_list.append(PosErr)

        # # ## Compute Body Angle Errors
        pred_bodyNorm = normalize(bodyNormalData[0],axis=0)
        bodyOriErr = (pred_bodyNorm - bodyNormalData[1][:,:pred_bodyNorm.shape[1]])**2           #pred_pos: (2,frames)
        bodyOriErr = np.sqrt(np.sum(bodyOriErr, axis=0))
        bodyOriErr_list.append(bodyOriErr)

        # # # ## Compute Face Angle Errors
        pred_faceNorm = normalize(faceNormalData[0],axis=0)
        faceOriErr = (pred_faceNorm - faceNormalData[1][:,:pred_faceNorm.shape[1]])**2           #pred_pos: (2,frames)
        faceOriErr = np.sqrt(np.sum(faceOriErr, axis=0))
        faceOriErr_list.append(faceOriErr)


        # ## Compute Face Angle Errors
        if bVisualize==False:
            continue

        glViewer.setPosOnly(posData)
        glViewer.setFaceNormal(faceNormalData)
        glViewer.setBodyNormal(bodyNormalData)

        # glViewer.init_gl()

        """Generate Trajectory in Holden's form by pos and body orientation"""
        #traj_list, initTrans_list,initRot_list = ConvertTrajectory_velocityForm(posData,bodyNormalData)
        #traj_list, initTrans_list,initRot_list = ConvertTrajectory_velocityForm([posData[0]],[bodyNormalData[0]])
        #glViewer.set_Holden_Trajectory_3(traj_list, initTrans=initTrans_list, initRot=initRot_list)
        glViewer.init_gl()
        
        # traj_list_seq.append(np.array(traj_list))



##Draw Error Figure
avg_posErr_list=[]
total_avg_posErr = 0
cnt=0
for p in posErr_list:
    avgValue = np.mean(p)
    avg_posErr_list.append(avgValue)
    print(avgValue)
    total_avg_posErr += avgValue*len(p)
    cnt += len(p)

total_avg_posErr = total_avg_posErr/cnt
print("total_avg_posErr: {}".format(total_avg_posErr))


##Draw Error Figure
avg_bodyOriErr_list=[]
total_avg_bodyOriErr = 0
cnt=0
for p in bodyOriErr_list:
    avgValue = np.mean(p)
    avg_bodyOriErr_list.append(avgValue)
    #print(avgValue)
    total_avg_bodyOriErr += avgValue*len(p)
    cnt += len(p)

total_avg_bodyOriErr = total_avg_bodyOriErr/cnt
print("total_avg_bodyOriErr: {}".format(total_avg_bodyOriErr))


##Draw Error Figure
avg_faceOriErr_list=[]
total_avg_faceOriErr = 0
cnt=0
for p in faceOriErr_list:
    avgValue = np.mean(p)
    avg_faceOriErr_list.append(avgValue)
    #print(avgValue)
    total_avg_faceOriErr += avgValue*len(p)
    cnt += len(p)

total_avg_faceOriErr = total_avg_faceOriErr/cnt
print("total_avg_faceOriErr: {}".format(total_avg_faceOriErr))


# add a subplot with no frame
bShowGraph = False
if bShowGraph:
    import matplotlib.pyplot as plt
    ax2=plt.subplot(311)
    plt.plot(avg_posErr_list)
    plt.title('Average Pos Error')

    ax2=plt.subplot(312)
    plt.plot(avg_bodyOriErr_list)
    plt.title('Average bodyOri Error')

    ax2=plt.subplot(313)
    plt.plot(avg_faceOriErr_list)
    plt.title('Average faceOri Error')
    plt.show()




