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
test_dblist = ['data_hagglingSellers_speech_formation_bySequence_white_brl_training_4fcn_atten']   #no normalized
test_dblist = ['data_hagglingSellers_speech_formation_bySequence_white_brl_testing_4fcn_atten']   #no normalized

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

bAblationStudy = False

######################################
# Checkout Folder and pretrain file setting
checkpointRoot = './'
checkpointFolder = checkpointRoot+ '/social_regressor_fcn/'
preTrainFileName= 'checkpoint_e50_loss0.0884.pth'

checkpointFolder = checkpointRoot+ '/social_regressor_fcn_try1/'
preTrainFileName= 'checkpoint_e4900_loss0.0867.pth'

checkpointFolder = checkpointRoot+ '/social_regressor_fcn_try6/'
preTrainFileName= 'checkpoint_e150_loss18.9973.pth'

checkpointFolder = checkpointRoot+ '/social_regressor_fcn_try8/'
preTrainFileName= 'checkpoint_e2350_loss15.2087.pth'

checkpointFolder = checkpointRoot+ '/social_regressor_fcn_try17/'
preTrainFileName= 'checkpoint_e1100_loss10.9647.pth'


#Normalized by the first frame
checkpointFolder = checkpointRoot+ '/social_regressor_fcn/'
preTrainFileName= 'checkpoint_e102_loss19.9177.pth'


#No normalization version with wrong std
checkpointFolder = checkpointRoot+ '/best/social_regressor_fcn_noNorm_badStd/'
preTrainFileName= 'checkpoint_e167_loss19.4441.pth'


#No normalization version with wrong std
checkpointFolder = checkpointRoot+ '/best/social_regressor_fcn_try19_noNorm_badStd/'
preTrainFileName= 'checkpoint_e102_loss19.7454.pth'


#No normalization version. no regularization
checkpointFolder = checkpointRoot+ '/best/social_regressor_fcn_try6_noNorm_noReg/'
preTrainFileName= 'checkpoint_e150_loss33.4851.pth'

#layer 3
checkpointFolder = checkpointRoot+ '/social_regressor_fcn_3_try2/'
preTrainFileName= 'checkpoint_e50_loss22.8530.pth'
preTrainFileName= 'checkpoint_e12000_loss29.2500.pth'
 




#Ablation study: position + BodyOri
bAblationStudy = True
checkpointFolder = checkpointRoot+ '/social_regressor_fcn_try1/'
preTrainFileName= 'checkpoint_e3_loss18.9841.pth'
mask = (2,3,  8,9)

#Ablation study: position + FaceOri
bAblationStudy = True
checkpointFolder = checkpointRoot+ '/social_regressor_fcn_try2_posFace/'
preTrainFileName= 'checkpoint_e21_loss18.4919.pth'
mask = (4,5, 10,11)

#Ablation study: position only
bAblationStudy = True
checkpointFolder = checkpointRoot+ '/social_regressor_fcn_try1_posOnly/'
preTrainFileName= 'checkpoint_e20_loss25.2516.pth'
mask = (2,3,4,5, 8,9,10,11)


#Best: Error: 24.594 No normalization version. with l1 regularization
checkpointFolder = checkpointRoot+ '/best/social_regressor_fcn_try6_noNorm_l1Reg/'
preTrainFileName= 'checkpoint_e45_loss17.9812.pth'
bAblationStudy = False

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
        test_attention = test_attention_all[seqIdx]
        test_sppech_raw = test_Y_raw_all[seqIdx]     #(3, frames)


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

        if bAblationStudy:
            # """Pos only"""
            # mask = (2,3,4,5, 8,9,10,11)
            # """Pos + face ori only"""
            # #mask = (4,5, 10,11)
            # """Pos + body ori only"""
            # #mask = (2,3,  8,9)
            
            test_X[:,mask,:] = preprocess['Xmean'][:, mask,:]
            


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

        gt_attention_face = test_attention[targetHumanIdx,:,0]  #(frame,)
        gt_attention_body = test_attention[targetHumanIdx,:,1] #(frame,)

        vis_data_input_b_pos = inputData_np_ori[:2,:]  #0,1 for position
        vis_data_input_b_faceNorm = inputData_np_ori[2:4,:]  #0,1 for position
        vis_data_input_b_bodyNorm = inputData_np_ori[4:6,:]  #0,1 for position

        attention_face_b = test_attention[0,:,0] #(frame,)
        attention_body_b = test_attention[0,:,1] #(frame,)
        
        # vis_data_input_1_faceAttention = test_attention[0,:,0]
        # vis_data_input_1_bodyAttention = test_attention[0,:,1]

        vis_data_input_r_pos = inputData_np_ori[6:8,:]  #0,1 for position
        vis_data_input_r_faceNorm = inputData_np_ori[8:10,:]  #0,1 for position
        vis_data_input_r_bodyNorm = inputData_np_ori[10:12,:]  #0,1 for position

        attention_face_r = test_attention[otherIdx,:,0] #(frame,)
        attention_body_r = test_attention[otherIdx,:,1] #(frame,)

        # vis_data_input_2_faceAttention = test_attention[0,:,0]
        # vis_data_input_2_bodyAttention = test_attention[0,:,1]
    
        posData = [pred_pos, vis_gt_pos, vis_data_input_b_pos, vis_data_input_r_pos]
        faceNormalData = [pred_faceNorm, vis_gt_faceNorm, vis_data_input_b_faceNorm, vis_data_input_r_faceNorm]
        bodyNormalData = [pred_bodyNorm, vis_gt_bodyNorm, vis_data_input_b_bodyNorm, vis_data_input_r_bodyNorm]

        #Make sure the frame lengths are the same
        frameLen = pred_pos.shape[1]
        posData = [ p[:,:frameLen] for p in posData]
        faceNormalData = [ p[:,:frameLen] for p in faceNormalData]
        bodyNormalData = [ p[:,:frameLen] for p in bodyNormalData]
        gt_attention_face = gt_attention_face[:frameLen]
        gt_attention_body = gt_attention_body[:frameLen]

        """Convert attention to direction"""
        """
        import attention
        #Debug: compute face normal from attention and put that as body normal
        leftPos = posData[1]    #target GT
        buyerPos = posData[2]
        rightPos = posData[3]
        faceNormalData[1] = attention.attention2Direction(leftPos, buyerPos, rightPos, gt_attention_body)
        faceNormalData[2] = attention.attention2Direction(buyerPos,rightPos, leftPos, attention_body_b)
        faceNormalData[3] = attention.attention2Direction(rightPos, leftPos, buyerPos, attention_body_r)

        bodyNormalData[1] = attention.attention2Direction(leftPos, buyerPos, rightPos, gt_attention_face)
        bodyNormalData[2] = attention.attention2Direction(buyerPos,rightPos, leftPos, attention_face_b)
        bodyNormalData[3] = attention.attention2Direction(rightPos, leftPos, buyerPos, attention_face_r)
        """

        # ####################################
        # ## Compute Errors
        PosErr = (pred_pos - vis_gt_pos[:,:pred_pos.shape[1]])**2           #pred_pos: (2,frames)
        PosErr = np.sqrt(np.sum(PosErr, axis=0))
        posErr_list.append(PosErr)

        # ## Compute Body Angle Errors
        pred_bodyNorm = normalize(pred_bodyNorm,axis=0)
        bodyOriErr = (pred_bodyNorm - vis_gt_bodyNorm[:,:pred_bodyNorm.shape[1]])**2           #pred_pos: (2,frames)
        bodyOriErr = np.sqrt(np.sum(bodyOriErr, axis=0))
        bodyOriErr_list.append(bodyOriErr)

        # ## Compute Face Angle Errors
        pred_faceNorm = normalize(pred_faceNorm,axis=0)
        faceOriErr = (pred_faceNorm - vis_gt_faceNorm[:,:pred_faceNorm.shape[1]])**2           #pred_pos: (2,frames)
        faceOriErr = np.sqrt(np.sum(faceOriErr, axis=0))
        faceOriErr_list.append(faceOriErr)


        # ## Compute Face Angle Errors
        if bVisualize==False:
            continue


        """Compute turn changing time"""
        speechSig = test_sppech_raw[2,:]    
        turnChange = np.where(abs(speechSig[1:] - speechSig[:-1] ) >0.5)[0]
        """Show only turn change time"""
        frameLeng = posData[0].shape[1]
        print(turnChange)
        selectedFrames = []
        for f in turnChange:
            fStart = max(f - 90,0)
            fEnd =  min(f + 90,frameLeng-1)
            selectedFrames += range(fStart,fEnd)

        for i in range(len(posData)):
            posData[i] = posData[i][:,selectedFrames]
            faceNormalData[i] = faceNormalData[i][:,selectedFrames]
            bodyNormalData[i] = bodyNormalData[i][:,selectedFrames]
            

        glViewer.resetFrameLimit()
        glViewer.setPosOnly(posData)
        glViewer.setFaceNormal(faceNormalData)
        glViewer.setBodyNormal(bodyNormalData)

        # glViewer.init_gl()

        # """Generate Trajectory in Holden's form by pos and body orientation"""
        # #traj_list, initTrans_list,initRot_list = ConvertTrajectory_velocityForm(posData,bodyNormalData)
        # traj_list, initTrans_list,initRot_list = ConvertTrajectory_velocityForm([posData[0]],[bodyNormalData[0]])
        # glViewer.set_Holden_Trajectory_3(traj_list, initTrans=initTrans_list, initRot=initRot_list)
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
std = np.std(avg_posErr_list)
print("total_avg_posErr: {0:.2f}, std {1:.2f}".format(total_avg_posErr, std))


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
std = np.std(avg_bodyOriErr_list)
print("total_avg_bodyOriErr: {0:.2f}, std {1:.2f}".format(total_avg_bodyOriErr,std))


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
std = np.std(avg_faceOriErr_list)
print("total_avg_faceOriErr: {0:.2f}, std {1:.2f}".format(total_avg_faceOriErr,std))


#save current values
output = open('predForm_1112_noNorm.pkl', 'wb')
pickle.dump({'avg_posErr_list':avg_posErr_list, 'avg_bodyOriErr_list':avg_bodyOriErr_list, 'avg_faceOriErr_list': avg_faceOriErr_list}, output)
output.close()



# #load current values
# pkl_file = open('predForm_1112_noNorm.pkl', 'rb')
# data = pickle.load(pkl_file)
# pkl_file.close()
# avg_posErr_list = data['avg_posErr_list']
# avg_bodyOriErr_list = data['avg_bodyOriErr_list']
# avg_faceOriErr_list = data['avg_faceOriErr_list']



# add a subplot with no frame
bShowGraph = True
if bShowGraph:
    import matplotlib.pyplot as plt
    plt.rc('xtick', labelsize=18)     
    plt.rc('ytick', labelsize=18)
    

    ax2=plt.subplot(311)
    plt.plot(avg_posErr_list)
    plt.title('Average Pos Error', fontsize=20)
    plt.grid()
    plt.xlabel('Seq. Index', fontsize=20)
    plt.ylabel('Error (cm)', fontsize=20)
    

    ax2=plt.subplot(312)
    plt.plot(avg_bodyOriErr_list)
    plt.title('Average body Orientation Error', fontsize=20)
    plt.grid()
    #plt.xlabel('Seq. Index', fontsize=15)
    plt.ylabel('Error (cm)', fontsize=20)

    ax2=plt.subplot(313)
    plt.plot(avg_faceOriErr_list)
    plt.title('Average face Orientation Error', fontsize=20)
    plt.grid()
    #plt.xlabel('Seq. Index', fontsize=15)
    plt.ylabel('Error (cm)', fontsize=20)
    

    plt.tight_layout()
    plt.show()




