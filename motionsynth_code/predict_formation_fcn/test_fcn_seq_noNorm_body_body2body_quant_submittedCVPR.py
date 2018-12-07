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

sys.path.append('/ssd/codes/pytorch_motionSynth/motionsynth_data/motion')
from Pivots import Pivots
from Quaternions import Quaternions

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
test_dblist = ['data_hagglingSellers_speech_formation_bySequence_white_brl_training_4fcn_atten']   #no normalized
test_dblist = ['data_hagglingSellers_speech_formation_bySequence_white_brl_testing_4fcn_atten_tiny']   #normalized
test_dblist = ['data_hagglingSellers_speech_formation_bySequence_white_brl_testing_4fcn']   #no normalized


test_dblist_body = ['data_hagglingSellers_speech_body_bySequence_white_noGa_brl_testing_tiny']
test_dblist_body = ['data_hagglingSellers_speech_body_bySequence_white_noGa_brl_testing']



"""Load formation data"""
pkl_file = open(datapath + test_dblist[0] + '.pkl', 'rb')
test_data = pickle.load(pkl_file)
pkl_file.close()

test_X_raw_all = test_data['data']  #Input (1044,240,73)
speech_raw_all = test_data['speech']  #Input (1044,240,73)
test_seqNames = test_data['seqNames']


"""Load body motion data"""
pkl_file_body = open(datapath + test_dblist_body[0] + '.pkl', 'rb')
test_data_body = pickle.load(pkl_file_body)
pkl_file_body.close()

test_body_raw_all = test_data_body['data']  #Input (1044,240,73)
test_body_seqNames = test_data_body['seqNames']
test_body_initInfo = test_data_body['initInfo']

 


######################################
# Verifying whether formation and body data are the same sequences
assert(len(test_seqNames) == len(test_body_seqNames))
for i,n in enumerate(test_body_seqNames):
    n2 = os.path.basename(test_seqNames[i])
    #print('{} vs {}'.format(n2,n))
    if False==(n2[:-4] in n):
        assert(False)


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
## Import Body2body Model
import modelZoo_body2body
checkpointRoot = './'
checkpointFolder_body2body = checkpointRoot+ '/best/social_regressor_fcn_bn_encoder_2_try2_body2body/'
preTrainFileName= 'checkpoint_e1000_loss0.1450.pth'
model_body2body = modelZoo_body2body.regressor_fcn_bn_encoder_2().cuda()


checkpointFolder_body2body = checkpointRoot+ '/body2body/social_regressor_fcn_bn_encoder_try2/'
preTrainFileName= 'checkpoint_e900_loss0.1382.pth'
model_body2body = modelZoo_body2body.regressor_fcn_bn_encoder().cuda()

preprocess_body2body = np.load(checkpointFolder_body2body + 'preprocess_core.npz') #preprocess['Xmean' or 'Xstd']: (1, 73,1)

#model = getattr(modelZoo_traj2Body,args.model)().cuda()
model_body2body.eval()

#Create Model
trainResultName = checkpointFolder_body2body + preTrainFileName
loaded_state = torch.load(trainResultName, map_location=lambda storage, loc: storage)

model_body2body.load_state_dict(loaded_state,strict=False) #strict False is needed some version issue for batchnorm
model_body2body = model_body2body.eval()  #Do I need this again?

b2b_body_mean = preprocess_body2body['body_mean']
b2b_body_std = preprocess_body2body['body_std']

b2b_body_mean_two = preprocess_body2body['body_mean_two']
b2b_body_std_two = preprocess_body2body['body_std_two']


############################################################################
# Import Pretrained Autoencoder

######################################
# Checkout Folder and pretrain file setting
ae_checkpointRoot = './'
ae_checkpointFolder = ae_checkpointRoot+ '/body2body/social_autoencoder_first_try9_120frm_best_noReg/'
preTrainFileName= 'checkpoint_e1009_loss0.0085.pth'


# ######################################
# # Load Pretrained Auto-encoder
ae_preprocess = np.load(ae_checkpointFolder + 'preprocess_core.npz') #preprocess['Xmean' or 'Xstd']: (1, 73,1))
model_ae_body = modelZoo_body2body.autoencoder_first().cuda()

#Creat Model
trainResultName = ae_checkpointFolder + preTrainFileName
loaded_state = torch.load(trainResultName, map_location=lambda storage, loc: storage)

model_ae_body.load_state_dict(loaded_state,strict=False) #strict False is needed some version issue for batchnorm
model_ae_body = model_ae_body.eval()  #Do I need this again?



############################
## Choose a sequence
#seqIdx =1
posErr_list = []
traj_list_seq =[]
traj2body_skeletonErr_list = []
body2body_skeletonErr_list = []
trajbody2body_skeletonErr_list =[]
bVisualize = True
for seqIdx in range(len(test_X_raw_all)):

    # if seqIdx!=len(test_X_raw_all)-1:
    #     continue

    for iteration in [1]:#[0,1]:  

        seqName = os.path.basename(test_seqNames[seqIdx])
        #if not ('170221_haggling_b2_group4' in seqName):
        # if not ('170221_haggling_b1_group4' in seqName):
        #     continue

        #Rendering Model
        outputFolder = '/ssd/render_ssp/{}'.format(seqName[:-4])
        if os.path.exists(outputFolder) == False:
            os.mkdir(outputFolder)
        else:
            continue

        print('{}-{}'.format(seqName, iteration))


        if iteration ==0:
            targetHumanIdx =1
            otherIdx =2
        else:
            targetHumanIdx =2
            otherIdx =1

        traj_X_raw = test_X_raw_all[seqIdx]     #(3, frames, feature:9)
        speech_raw = speech_raw_all[seqIdx]     #(3, frames)
        body_raw_group = test_body_raw_all[seqIdx]    #(3, frames, features:73)

        features = (0,2, 3,5, 6,8)  #Ignoring Y axis
        traj_X_raw = traj_X_raw[:,:,features]       #(3, frames, features:6)   #2D location only ignoring Y

        traj_X = traj_X_raw[0,:,:]      #(frames, features:6) //buyer
        traj_X = np.concatenate( (traj_X, traj_X_raw[otherIdx,:,:]), axis= 1)      #(frames, features: 12) [buyer;seller1]
        #test_X = test_X[:,(0,1, 4,5)] #Pos only
        traj_Y_GT = traj_X_raw[targetHumanIdx,:,:]    #(frames, features:6) #Target human

        traj_X = np.swapaxes(np.expand_dims(traj_X,0),1,2).astype(np.float32)  #(1, frames,feature:12)
        traj_Y_GT = np.swapaxes(np.expand_dims(traj_Y_GT,0),1,2).astype(np.float32)  #(1, frames,feature:6)

        traj_X_stdd = (traj_X[:,:] - preprocess['Xmean']) / preprocess['Xstd']   #(frames, features: 12) [buyer;seller1]

        traj_Ymean = preprocess['Ymean']
        traj_Ystd = preprocess['Ystd']

        ######################################
        # Testing
        pred_all = np.empty([0,1],dtype=float)
        traj_X_vis =np.empty([0,200],dtype=float)

        #idxStart  = bi*batch_size
        idxStart  = 0#bi*batch_size
        batch_size = traj_X.shape[0]

        """ Predicting Trajectory of the target person """
        inputData = Variable(torch.from_numpy(traj_X_stdd)).cuda()
        outputGT = Variable(torch.from_numpy(traj_Y_GT)).cuda()

        predic_traj = model(inputData)

        predic_traj = predic_traj.data.cpu().numpy() #(batch, feature:6, frames)
        predic_traj = predic_traj*traj_Ystd + traj_Ymean   #(batch, feature:6, frames)
        predic_traj = predic_traj[0, :,:] #(feature:6, frames)

        #traj_X = traj_X[0,:,:] #(feature:12, frames)
        #vis_gt = outputData_np[0,:,:]  #(feature:6, frames)

        """Generate visualization formats"""
        #Prediction
        pred_pos = predic_traj[:2,:]
        pred_faceNorm = predic_traj[2:4,:]
        pred_bodyNorm = predic_traj[4:6,:]

        #Ground Truth of the output
        traj_Y_GT =traj_Y_GT[0]
        targetP_gt_pos= traj_Y_GT[:2,:]
        targetP_faceNorm= traj_Y_GT[2:4,:]
        targetP_bodyNorm= traj_Y_GT[4:6,:]

        #Input
        traj_X = traj_X[0]  #(features:12, frames)
        gt_input_1_pos = traj_X[:2,:]  #0,1 for position
        gt_input_1_faceNorm = traj_X[2:4,:]  #0,1 for position
        gt_input_1_bodyNorm = traj_X[4:6,:]  #0,1 for position

        gt_input_2_pos = traj_X[6:8,:]  #0,1 for position
        gt_input_2_faceNorm = traj_X[8:10,:]  #0,1 for position
        gt_input_2_bodyNorm = traj_X[10:12,:]  #0,1 for position
    
        vis_posData = [pred_pos, targetP_gt_pos, gt_input_1_pos, gt_input_2_pos]
        vis_faceNormalData = [pred_faceNorm, targetP_faceNorm, gt_input_1_faceNorm, gt_input_2_faceNorm]
        vis_bodyNormalData = [pred_bodyNorm, targetP_bodyNorm, gt_input_1_bodyNorm, gt_input_2_bodyNorm]
       

        """Generate Trajectory in Holden's form by pos and body orientation"""
        traj_list_holdenForm, pred_initTrans_list, pred_initRot_list = utility.ConvertTrajectory_velocityForm( [pred_pos], [pred_bodyNorm])       #Prediction version
        gt_traj_list_holdenForm, gt_initTrans_list, gt_initRot_list = utility.ConvertTrajectory_velocityForm(vis_posData[1:2], vis_bodyNormalData[1:2])       #GT version
        #traj_list, initTrans_list, initRot_list = utility.ConvertTrajectory_velocityForm(posData[1:2], bodyNormalData[1:2])      #GT version
        #glViewer.set_Holden_Trajectory_3(traj_list, initTrans=initTrans_list, initRot=initRot_list)
        # glViewer.init_gl()

        """ Apply Traj2Body """
        test_traj = traj_list_holdenForm[0] #(3, frames)
        test_traj = np.expand_dims(test_traj,0).astype(np.float32) #(num, 3, frameNum)

        ## Standardization
        test_traj_std = (test_traj - tj2body_traj_mean) / tj2body_traj_std

        inputData_np = test_traj_std
        inputData = Variable(torch.from_numpy(inputData_np)).cuda()  #(batch, 3, frameNum)
        output = model_traj2body(inputData)

        output_body_np = output.data.cpu().numpy()  #(batch, 73, frames)
        output_body_np = output_body_np[:,:69,:]      #crop the last 4, if there exists
        
        #Original code
        output_body_np = output_body_np*tj2body_body_std[:,:-4,:] + tj2body_body_mean[:,:-4,:]
        
        ##Quant: Baseline. Only Mean
        #output_body_np = output_body_np*tj2body_body_std[:,:-4,:]*0.0 + tj2body_body_mean[:,:-4,:]

       
        ## Optional: Overwrite global trans oreintation info
        output_body_np[:,-3:,:] =  test_traj[:,:,:output_body_np.shape[2]]         

        output_body_np = np.swapaxes(output_body_np,1,2)  #(batch, frames, 73)
        output_body_np = np.reshape(output_body_np,(-1,69))
        pred_traj2body = np.swapaxes(output_body_np,0,1)

        """For visualization: Get GT Body with global infomation. Order:Target buyer seller1)"""
        vis_bodyData = [ body_raw_group[2,:,:], body_raw_group[0,:,:], body_raw_group[1,:,:] ]  #(frames, 73)
        for i,X in enumerate(vis_bodyData):
            vis_bodyData[i] =  np.swapaxes(X, 0, 1).astype(np.float32) #(73, frames)

        vis_bodyGT_initTrans = [test_body_initInfo[seqIdx][2]['pos'], test_body_initInfo[seqIdx][0]['pos'], test_body_initInfo[seqIdx][1]['pos'] ]
        vis_bodyGT_initRot = [test_body_initInfo[seqIdx][2]['rot'], test_body_initInfo[seqIdx][0]['rot'], test_body_initInfo[seqIdx][1]['rot']]
        vis_bodyGT_initRot = [ Quaternions(x) for x in vis_bodyGT_initRot ]


        # ####################################
        # ## Compute Skeleton Error
        #Only consider the predicted on
        HOLDEN_DATA_SCALING = 5
        bodyData_pred = pred_traj2body[:-3,:]*HOLDEN_DATA_SCALING   #66,frames
        bodyData_gt = vis_bodyData[0][:-7,:]*HOLDEN_DATA_SCALING       #66, frames
        bodyData_gt = bodyData_gt[:,:bodyData_pred.shape[1]]
        skelErr = ( bodyData_pred -  bodyData_gt)**2           #66, frames
        skelErr = np.reshape(skelErr, (3,22,-1))        #3,22, frames
        skelErr = np.sqrt(np.sum(skelErr, axis=0))      #22,frames        
        skelErr = np.mean(skelErr,axis=0)   #frames
        traj2body_skeletonErr_list.append(skelErr)


        """Apply Body2Body"""
        ## Test data
        input_body = body_raw_group[0:1,:,:].copy()      #(1, frames, features:73) //person0,1's all values (position, head orientation, body orientation)
        input_body = np.concatenate( (input_body, body_raw_group[1:2,:,:].copy()), axis= 2)      #(1, frames, features:146)
        output_body_GT = body_raw_group[2:3,:,:].copy()    #(1, frames, features:73)

        ######################################
        # Data pre-processing
        input_body = np.swapaxes(input_body, 1, 2).astype(np.float32)   #(1, features, frames)
        output_body_GT = np.swapaxes(output_body_GT, 1, 2).astype(np.float32)   #(1, features, frames)

        ######################################
        # Data pre-processing
        test_X_std = (input_body - b2b_body_mean_two) / b2b_body_std_two
        test_Y_std = (output_body_GT - b2b_body_mean) / b2b_body_std

        inputData = Variable(torch.from_numpy(test_X_std)).cuda()  #(batch, 3, frameNum)

        # ===================forward=====================
        output = model_body2body(inputData)
        output = model_ae_body.decoder(output)
        #loss = criterion(output, outputGT)
        #loss = criterion(output, outputGT)


        #De-standardaize
        output_np = output.data.cpu().numpy()  #(batch, 73, frames)
        output_np = output_np*b2b_body_std + b2b_body_mean

        output_np = np.swapaxes(output_np,1,2)  #(batch, frames, 73)
        output_np = np.reshape(output_np,(-1,73))
        pred_body2body = np.swapaxes(output_np,0,1)  #(73, frames)
        pred_body2body = pred_body2body[:69,:]    #(69, frames)


        # ## Compute Skeleton Error
        #Only consider the predicted on
        HOLDEN_DATA_SCALING = 5
        bodyData_pred = pred_body2body[:-3,:]*HOLDEN_DATA_SCALING   #66,frames
        bodyData_gt = vis_bodyData[0][:-7,:]*HOLDEN_DATA_SCALING       #66, frames
        bodyData_gt = bodyData_gt[:,:bodyData_pred.shape[1]]
        skelErr = ( bodyData_pred -  bodyData_gt)**2           #66, frames
        skelErr = np.reshape(skelErr, (3,22,-1))        #3,22, frames
        skelErr = np.sqrt(np.sum(skelErr, axis=0))      #22,frames        
        skelErr = np.mean(skelErr,axis=0)   #frames
        body2body_skeletonErr_list.append(skelErr)

        #Add this guy for vis Unit
        # Visualize: traj2body body2body GTTarget input1 input2
        # vis_bodyData = [ pred_traj2body, pred_body2body] + vis_bodyData
        # vis_bodyGT_initRot = [ vis_bodyGT_initRot[0], vis_bodyGT_initRot[0]] + vis_bodyGT_initRot
        # vis_bodyGT_initTrans = [ vis_bodyGT_initTrans[0], vis_bodyGT_initTrans[0]] +  vis_bodyGT_initTrans
        vis_bodyData = [ pred_traj2body, pred_body2body] + vis_bodyData
        vis_bodyGT_initRot = [ pred_initRot_list[0], pred_initRot_list[0]] + vis_bodyGT_initRot
        vis_bodyGT_initTrans = [ pred_initTrans_list[0], pred_initTrans_list[0]] +  vis_bodyGT_initTrans


        """Set the same length"""
        frameLeng = pred_traj2body.shape[1]
        for i,v in enumerate(vis_bodyData):
            vis_bodyData[i] = vis_bodyData[i][:,:frameLeng]

        """ Substitute the leg of body2body by traj2body"""
        vis_bodyData[1][0:33,:] = vis_bodyData[0][:33,:]
        vis_bodyData[1][-3:,:] = vis_bodyData[0][-3:,:]


        # ## Compute Skeleton Error
        #Only consider the predicted on
        HOLDEN_DATA_SCALING = 5
        bodyData_pred = vis_bodyData[1][:-3,:]*HOLDEN_DATA_SCALING   #66,frames
        bodyData_gt = vis_bodyData[2][:-7,:]*HOLDEN_DATA_SCALING       #66, frames
        bodyData_gt = bodyData_gt[:,:bodyData_pred.shape[1]]
        skelErr = ( bodyData_pred -  bodyData_gt)**2           #66, frames
        skelErr = np.reshape(skelErr, (3,22,-1))        #3,22, frames
        skelErr = np.sqrt(np.sum(skelErr, axis=0))      #22,frames        
        skelErr = np.mean(skelErr,axis=0)   #frames
        trajbody2body_skeletonErr_list.append(skelErr)

        
        #Final slection to draw a subset
        #GT: GTTarget, inputBuyer, inputSeller
        # vis_bodyData = [ vis_bodyData[2], vis_bodyData[3], vis_bodyData[4] ]
        # vis_bodyGT_initRot = [ vis_bodyGT_initRot[2], vis_bodyGT_initRot[3], vis_bodyGT_initRot[4] ] 
        # vis_bodyGT_initTrans = [ vis_bodyGT_initTrans[2], vis_bodyGT_initTrans[3], vis_bodyGT_initTrans[4] ]


        #Traj2Body, inputBuyer, inputSeller
        #vis_bodyData = [ vis_bodyData[0], vis_bodyData[3], vis_bodyData[4] ]
        vis_bodyData = [ vis_bodyData[1], vis_bodyData[3], vis_bodyData[4] ]
        vis_bodyGT_initRot = [ vis_bodyGT_initRot[0], vis_bodyGT_initRot[3], vis_bodyGT_initRot[4] ] 
        vis_bodyGT_initTrans = [ vis_bodyGT_initTrans[0], vis_bodyGT_initTrans[3], vis_bodyGT_initTrans[4] ]


        """ Select Trajectory to visualize """
        #GT: GTTargetTrajectory + inputBuyer, inputSeller
        # vis_posData = [vis_posData[1], vis_posData[2], vis_posData[3] ]
        # vis_faceNormalData = [vis_faceNormalData[1], vis_faceNormalData[2], vis_faceNormalData[3] ]
        # vis_bodyNormalData = [vis_bodyNormalData[1], vis_bodyNormalData[2], vis_bodyNormalData[3] ]

        #Pred trajectory + inputBuyer, inputSeller
        vis_posData = [vis_posData[0], vis_posData[2], vis_posData[3] ]
        vis_faceNormalData = [vis_faceNormalData[0], vis_faceNormalData[2], vis_faceNormalData[3] ]
        vis_bodyNormalData = [vis_bodyNormalData[0], vis_bodyNormalData[2], vis_bodyNormalData[3] ]

        if bVisualize==False:
            continue

        """Visualize Location + Orientation"""
        glViewer.setPosOnly(vis_posData)
        glViewer.setFaceNormal(vis_faceNormalData)
        glViewer.setBodyNormal(vis_bodyNormalData)
        
        """Visualize Trajectory"""
        glViewer.set_Holden_Trajectory_3(traj_list_holdenForm, initTrans=pred_initTrans_list, initRot=pred_initRot_list)   #pred
        #glViewer.set_Holden_Trajectory_3(gt_traj_list_holdenForm, initTrans=gt_initTrans_list, initRot=gt_initRot_list)   #GT
        #glViewer.set_Holden_Trajectory_3([ bodyData[0][-7:-4,:], bodyData[1][-7:-4,:], bodyData[2][-7:-4,:] ], initRot=initRot, initTrans= initTrans)

        """Visualize Body"""
        #glViewer.set_Holden_Data_73([output_body_np],initTrans=initTrans_list,initRot=initRot_list)
        glViewer.set_Holden_Data_73(vis_bodyData, ignore_root=False, initRot=vis_bodyGT_initRot, initTrans= vis_bodyGT_initTrans, bIsGT=True)
        #glViewer.set_Holden_Data_73(vis_bodyData, ignore_root=False, initRot=vis_bodyGT_initRot, initTrans= vis_bodyGT_initTrans, bIsGT=True)

        
        glViewer.setSaveOnlyMode(True)
        glViewer.setSave(True)
        glViewer.setSaveFoldeName(outputFolder)
        glViewer.LoadCamViewInfo()


        glViewer.init_gl()

# Compute error


##Draw Error Figure
avg_skelErr_list=[]
total_avg_skelErr = 0
cnt=0
for p in traj2body_skeletonErr_list:
    avgValue = np.mean(p)
    avg_skelErr_list.append(avgValue)
    print(avgValue)
    total_avg_skelErr += avgValue*len(p)
    cnt += len(p)

total_avg_skelErr = total_avg_skelErr/cnt
std = np.std(avg_skelErr_list)
print("traj2body_skeletonErr_list: {}, std {}".format(total_avg_skelErr, std))

##Draw Error Figure
avg_skelErr_list=[]
total_avg_skelErr = 0
cnt=0
for p in body2body_skeletonErr_list:
    avgValue = np.mean(p)
    avg_skelErr_list.append(avgValue)
    print(avgValue)
    total_avg_skelErr += avgValue*len(p)
    cnt += len(p)

total_avg_skelErr = total_avg_skelErr/cnt
std = np.std(avg_skelErr_list)
print("body2body_skeletonErr_list: {}, std {}".format(total_avg_skelErr, std))

##Draw Error Figure
avg_skelErr_list=[]
total_avg_skelErr = 0
cnt=0
for p in trajbody2body_skeletonErr_list:
    avgValue = np.mean(p)
    avg_skelErr_list.append(avgValue)
    print(avgValue)
    total_avg_skelErr += avgValue*len(p)
    cnt += len(p)

total_avg_skelErr = total_avg_skelErr/cnt
std = np.std(avg_skelErr_list)
print("trajbody2body_skeletonErr_list: {}, std {}".format(total_avg_skelErr, std))


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




