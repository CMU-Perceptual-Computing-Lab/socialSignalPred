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

#from utility import data_2dTo3D
#from utility import ConvertTrajectory_velocityForm

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

test_dblist = ['data_hagglingSellers_speech_body_bySequence_white_noGa_brl_testing']   #no normalized
#test_dblist = ['data_hagglingSellers_speech_body_bySequence_white_noGa_brl_training']   #no normalized


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
checkpointFolder = checkpointRoot+ '/social_regressor_fcn_bn_test/'
preTrainFileName= 'checkpoint_e2_acc0.7455.pth'

#Submitted version
#Own body
checkpointRoot = '/posefs2b/Users/hanbyulj/pytorch_motionSynth/checkpoint/'
checkpointFolder = checkpointRoot+ '/body2speak_fcn_final/social_regressor_fcn_bn_dropout_own/'
preTrainFileName= 'checkpoint_e1_acc0.7593.pth'

#Submitted version
#Other body (using social)
checkpointRoot = '/posefs2b/Users/hanbyulj/pytorch_motionSynth/checkpoint/'
checkpointFolder = checkpointRoot+ '/body2speak_fcn_final/social_regressor_fcn_bn_dropout_social/'
preTrainFileName= 'checkpoint_e0_acc0.7059.pth'


#OWn Body
checkpointRoot = './'
checkpointFolder = checkpointRoot+ '/best_afterSubmit/social_regressor_fcn_bn_updated2_try1_own_8028/'
preTrainFileName= 'checkpoint_e367_acc0.8028.pth'
bOwnBody = True


#After submission
#Other body (using social)
checkpointRoot = './'
checkpointFolder = checkpointRoot+ '/best_afterSubmit/social_regressor_fcn_bn_updated2_social_7554/'
preTrainFileName= 'checkpoint_e1095_acc0.7554.pth'
bOwnBody = False


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
accuracy_sum = 0
cnt_sum=0
for seqIdx in range(len(test_X_raw_all)):

    # if seqIdx!=len(test_X_raw_all)-1:
    #     continue

    for iteration in [1]:#[0,1]:  

        seqName_base = os.path.basename(test_seqNames[seqIdx])
        # if bVisualize == True and not ('170228_haggling_b2_group1' in seqName_base):
        #     continue

        print('{}-{}'.format(seqName_base, iteration))
       
        test_X_raw = test_X_raw_all[seqIdx]     #(3, frames, feature:73)
        test_sppech_raw = test_Y_raw_all[seqIdx]     #(3, frames)

        # """Own Body"""
        if bOwnBody:
            test_X = test_X_raw[1,:,:]      #(num, 73) 
            frameLeng = test_X.shape[0]
            test_X = np.concatenate( (test_X, test_X_raw[2,:,:]), axis= 0)      #(num, 73)

            test_Y = test_sppech_raw[1]['indicator']
            test_Y = test_Y[:frameLeng]
            test_Y2 = test_sppech_raw[2]['indicator']
            test_Y2 = test_Y2[:frameLeng]
            test_Y = np.concatenate( (test_Y, test_Y2), axis= 0)
        else:
            """Other Subject Body (social)"""
            test_X = test_X_raw[1,:,:]      #(num, 73) 
            frameLeng = test_X.shape[0]
            test_X = np.concatenate( (test_X, test_X_raw[2,:,:]), axis= 0)      #(num, 73)

            test_Y = test_sppech_raw[2]['indicator']
            test_Y = test_Y[:frameLeng]
            test_Y2 = test_sppech_raw[1]['indicator']
            test_Y2 = test_Y2[:frameLeng]
            test_Y = np.concatenate( (test_Y, test_Y2), axis= 0)

        """Other Subject Body by shuffling (non-social)"""
        # #Select random sequence
        # idxCand = range(len(test_X_raw_all))
        # idxCand.remove(seqIdx)
        # rng.shuffle(idxCand)
        # randomIdx = idxCand[0]
        # print("randomIdx: {} vs ({})".format(randomIdx,seqIdx))
        # test_sppech_raw = test_Y_raw_all[randomIdx]     #(3, frames)

        # #Adjust length
        # frameLeng = min(test_sppech_raw[0]['indicator'].shape[0], test_X_raw.shape[1])
        # test_X = test_X_raw[1,:frameLeng,:]      #(num, 73) 
        # frameLeng = test_X.shape[0]
        # test_X = np.concatenate( (test_X, test_X_raw[2,:frameLeng,:]), axis= 0)      #(num, 73)

        # test_Y = test_sppech_raw[2]['indicator']
        # test_Y = test_Y[:frameLeng]
        # test_Y2 = test_sppech_raw[1]['indicator']
        # test_Y2 = test_Y2[:frameLeng]
        # test_Y = np.concatenate( (test_Y, test_Y2), axis= 0)

        
        ######################################
        # Standardization
        test_X = np.swapaxes(np.expand_dims(test_X,0),1,2).astype(np.float32)  #(1, frames,feature:12)
        test_Y = test_Y.astype(np.float32)
        test_X_stdd = (test_X[:,:] - preprocess['Xmean']) / preprocess['Xstd']

        #Ymean = preprocess['Ymean']
        #Ystd = preprocess['Ystd']

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
        pred = np.squeeze(pred)


        # ####################################
        # ## Compute Errors
        accuracy = (outputGT.eq( (output[0,0,:]>0.5).float() )).sum() #Just check the last one
        
        accuracy_sum += accuracy.item()
        cnt_sum +=len(pred)
        #accuracy = float(correct.item()) / len(outputData_np)
        #accuracy = sum(pred>0.5) == test_Y) / len(pred)
        print("accuracy: {}".format(float(accuracy)/len(pred)))

        # """ Plot output"""
        # import matplotlib.pyplot as plt
        # plt.plot(np.squeeze(pred))
        # plt.hold(True)
        # plt.plot(outputData_np)
        # plt.show()




        continue


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

acc_avg = float(accuracy_sum)/cnt_sum

print("acc_avg: {}".format(acc_avg))


# ##Draw Error Figure
# avg_posErr_list=[]
# total_avg_posErr = 0
# cnt=0
# for p in posErr_list:
#     avgValue = np.mean(p)
#     avg_posErr_list.append(avgValue)
#     print(avgValue)
#     total_avg_posErr += avgValue*len(p)
#     cnt += len(p)

# total_avg_posErr = total_avg_posErr/cnt
# std = np.std(avg_posErr_list)
# print("total_avg_posErr: {}, std {}".format(total_avg_posErr, std))


# ##Draw Error Figure
# avg_bodyOriErr_list=[]
# total_avg_bodyOriErr = 0
# cnt=0
# for p in bodyOriErr_list:
#     avgValue = np.mean(p)
#     avg_bodyOriErr_list.append(avgValue)
#     #print(avgValue)
#     total_avg_bodyOriErr += avgValue*len(p)
#     cnt += len(p)

# total_avg_bodyOriErr = total_avg_bodyOriErr/cnt
# std = np.std(avg_bodyOriErr_list)
# print("total_avg_bodyOriErr: {}, std {}".format(total_avg_bodyOriErr,std))


# ##Draw Error Figure
# avg_faceOriErr_list=[]
# total_avg_faceOriErr = 0
# cnt=0
# for p in faceOriErr_list:
#     avgValue = np.mean(p)
#     avg_faceOriErr_list.append(avgValue)
#     #print(avgValue)
#     total_avg_faceOriErr += avgValue*len(p)
#     cnt += len(p)

# total_avg_faceOriErr = total_avg_faceOriErr/cnt
# std = np.std(avg_faceOriErr_list)
# print("total_avg_faceOriErr: {}, std {}".format(total_avg_faceOriErr,std))


# #save current values
# output = open('predForm_1112_noNorm.pkl', 'wb')
# pickle.dump({'avg_posErr_list':avg_posErr_list, 'avg_bodyOriErr_list':avg_bodyOriErr_list, 'avg_faceOriErr_list': avg_faceOriErr_list}, output)
# output.close()



# # #load current values
# # pkl_file = open('predForm_1112_noNorm.pkl', 'rb')
# # data = pickle.load(pkl_file)
# # pkl_file.close()
# # avg_posErr_list = data['avg_posErr_list']
# # avg_bodyOriErr_list = data['avg_bodyOriErr_list']
# # avg_faceOriErr_list = data['avg_faceOriErr_list']



# # add a subplot with no frame
# bShowGraph = True
# if bShowGraph:
#     import matplotlib.pyplot as plt
#     plt.rc('xtick', labelsize=18)     
#     plt.rc('ytick', labelsize=18)
    

#     ax2=plt.subplot(311)
#     plt.plot(avg_posErr_list)
#     plt.title('Average Pos Error', fontsize=20)
#     plt.grid()
#     plt.xlabel('Seq. Index', fontsize=20)
#     plt.ylabel('Error (cm)', fontsize=20)
    

#     ax2=plt.subplot(312)
#     plt.plot(avg_bodyOriErr_list)
#     plt.title('Average body Orientation Error', fontsize=20)
#     plt.grid()
#     #plt.xlabel('Seq. Index', fontsize=15)
#     plt.ylabel('Error (cm)', fontsize=20)

#     ax2=plt.subplot(313)
#     plt.plot(avg_faceOriErr_list)
#     plt.title('Average face Orientation Error', fontsize=20)
#     plt.grid()
#     #plt.xlabel('Seq. Index', fontsize=15)
#     plt.ylabel('Error (cm)', fontsize=20)
    

#     plt.tight_layout()
#     plt.show()




