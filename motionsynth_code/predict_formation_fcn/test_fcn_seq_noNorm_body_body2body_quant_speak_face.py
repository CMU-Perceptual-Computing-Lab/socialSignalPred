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
# Sub networks
from network_face2face import Network_face2face 



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

test_dblist_body = ['data_hagglingSellers_speech_body_face_bySequence_white_noGa_brl_testing_tiny']
test_dblist_body = ['data_hagglingSellers_speech_body_face_bySequence_white_noGa_brl_testing']





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


"""Load face expression data"""
test_face_raw_all = test_data_body['face']  #Input (1044,240,73)

 


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
preprocess_traj = np.load(checkpointFolder + 'preprocess_core.npz') #preprocess['Xmean' or 'Xstd']: (1, 73,1)


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
## Import Body2speak Model
import modelZoo_body2speak
checkpointRoot = './'
#submitted
checkpointFolder_body2speak = checkpointRoot+ '/body2speak/social_regressor_fcn_bn_dropout_own/'
preTrainFileName= 'checkpoint_e1_acc0.7593.pth'

#submitted
checkpointFolder_body2speak = checkpointRoot+ '/body2speak/social_regressor_fcn_bn_dropout_social/'    
preTrainFileName= 'checkpoint_e0_acc0.7059.pth'


#after submission (best 75%)
checkpointFolder_body2speak = checkpointRoot+ '/body2speak/social_regressor_fcn_bn_updated2_social_7554/'    
preTrainFileName= 'checkpoint_e1095_acc0.7554.pth'


preprocess_body2speak = np.load(checkpointFolder_body2speak + 'preprocess_core.npz') #preprocess['Xmean' or 'Xstd']: (1, 73,1)

b2s_body_mean = preprocess_body2speak['Xmean']
b2s_body_std = preprocess_body2speak['Xstd']

#model = getattr(modelZoo_traj2Body,args.model)().cuda()
model_body2speak = modelZoo_body2speak.regressor_fcn_bn_dropout().cuda()
model_body2speak.eval()

#Create Model
trainResultName = checkpointFolder_body2speak + preTrainFileName
loaded_state = torch.load(trainResultName, map_location=lambda storage, loc: storage)

model_body2speak.load_state_dict(loaded_state,strict=False) #strict False is needed some version issue for batchnorm
model_body2speak = model_body2speak.eval()  #Do I need this again?


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


# ######################################
# # Load Face2Face
net_face2face = Network_face2face()



############################
## Choose a sequence
#seqIdx =1
posErr_list = []
traj_list_seq =[]
traj2body_skeletonErr_list = []
body2body_skeletonErr_list = []
trajbody2body_skeletonErr_list =[]
bVisualize = True
bRender = True         #IF true, save the opengl vis to files (/ssd/render_ssp/)
for seqIdx in range(len(test_X_raw_all)):

    seqName = os.path.basename(test_seqNames[seqIdx])
    #if not ('170221_haggling_b2_group4' in seqName):
    # if not ('170221_haggling_b1_group0' in seqName or 
    #             '170221_haggling_b1_group2' in seqName or 
    #             '170228_haggling_b1_group3' in seqName or
    #                 '170228_haggling_b1_group6' in seqName or
    #                     '170228_haggling_b1_group7' in seqName or
    #                     '170228_haggling_b2_group0' in seqName ) :
    #     continue
    #Rendering Model
    if bRender:
        outputFolder = '/ssd/render_ssp/{}'.format(seqName[:-4])
        if os.path.exists(outputFolder) == False:
            os.mkdir(outputFolder)
        else:
            continue

    print('{}'.format(seqName))

    targetSeller =2
    otherSeller =1

    #######################################################
    # All raw data
    traj_raw_group = test_X_raw_all[seqIdx]     #(3, frames, feature:9)
    speech_raw = speech_raw_all[seqIdx]     #(3, frames)
    body_raw_group = test_body_raw_all[seqIdx]    #(3, frames, features:73)
    face_raw_group = test_face_raw_all[seqIdx]    #(3, frames, features:73)


    #######################################################
    # Pre-processing Trajectory Data
    features = (0,2, 3,5, 6,8)  #Ignoring Y axis
    traj_raw_group = traj_raw_group[:,:,features]       #(3, frames, features:6)   #2D location only ignoring Y

    traj_in = traj_raw_group[0,:,:]      #(frames, features:6) //buyer
    traj_in = np.concatenate( (traj_in, traj_raw_group[otherSeller,:,:]), axis= 1)      #(frames, features: 12) [buyer;seller1]
    traj_out_GT = traj_raw_group[targetSeller,:,:]    #(frames, features:6) #Target human

    traj_in = np.swapaxes(np.expand_dims(traj_in,0),1,2).astype(np.float32)  #(1, frames,feature:12)
    traj_out_GT = np.swapaxes(np.expand_dims(traj_out_GT,0),1,2).astype(np.float32)  #(1, frames,feature:6)
    traj_in_stdd = (traj_in[:,:] - preprocess_traj['Xmean']) / preprocess_traj['Xstd']   #(frames, features: 12) [buyer;seller1]
    # traj_Ymean = preprocess_traj['Ymean']
    # traj_Ystd = preprocess_traj['Ystd']


    #######################################################
    # Predicting Trajectory Data
    inputData_ = Variable(torch.from_numpy(traj_in_stdd)).cuda()
    predic_traj = model(inputData_)
    predic_traj = predic_traj.data.cpu().numpy() #(batch, feature:6, frames)
    predic_traj = predic_traj * preprocess_traj['Ystd'] + preprocess_traj['Ymean']   #(batch, feature:6, frames)
    predic_traj = predic_traj[0, :,:] #(feature:6, frames)

    """Generate visualization formats"""
    #Prediction
    pred_pos = predic_traj[:2,:]
    pred_faceNorm = predic_traj[2:4,:]
    pred_bodyNorm = predic_traj[4:6,:]

    #Ground Truth of the output
    traj_out_GT =traj_out_GT[0]
    targetP_gt_pos= traj_out_GT[:2,:]
    targetP_faceNorm= traj_out_GT[2:4,:]
    targetP_bodyNorm= traj_out_GT[4:6,:]

    #Input
    traj_in = traj_in[0]  #(features:12, frames)
    gt_input_1_pos = traj_in[:2,:]  #0,1 for position
    gt_input_1_faceNorm = traj_in[2:4,:]  #0,1 for position
    gt_input_1_bodyNorm = traj_in[4:6,:]  #0,1 for position

    gt_input_2_pos = traj_in[6:8,:]  #0,1 for position
    gt_input_2_faceNorm = traj_in[8:10,:]  #0,1 for position
    gt_input_2_bodyNorm = traj_in[10:12,:]  #0,1 for position

    #Prediction, GT, input1, input2
    vis_traj_posData = [pred_pos, targetP_gt_pos, gt_input_1_pos, gt_input_2_pos]
    vis_traj_faceNormalData = [pred_faceNorm, targetP_faceNorm, gt_input_1_faceNorm, gt_input_2_faceNorm]
    vis_traj_bodyNormalData = [pred_bodyNorm, targetP_bodyNorm, gt_input_1_bodyNorm, gt_input_2_bodyNorm]
    

    #######################################################
    # Pre-processing Data for Traj2Body
    """Generate Trajectory in Holden's form by pos and body orientation"""
    traj_list_holdenForm, pred_initTrans_list, pred_initRot_list = utility.ConvertTrajectory_velocityForm( [pred_pos], [pred_bodyNorm])       #Prediction version
    gt_traj_list_holdenForm, gt_initTrans_list, gt_initRot_list = utility.ConvertTrajectory_velocityForm(vis_traj_posData[1:2], vis_traj_bodyNormalData[1:2])       #GT version
    #traj_list, initTrans_list, initRot_list = utility.ConvertTrajectory_velocityForm(posData[1:2], bodyNormalData[1:2])      #GT version
    #glViewer.set_Holden_Trajectory_3(traj_list, initTrans=initTrans_list, initRot=initRot_list)
    #glViewer.init_gl()

    """ Apply Traj2Body """
    t2body_in = traj_list_holdenForm[0] #(3, frames)
    t2body_in = np.expand_dims(t2body_in,0).astype(np.float32) #(num, 3, frameNum)
    ## Standardization
    t2body_in_std = (t2body_in - tj2body_traj_mean) / tj2body_traj_std

    #######################################################
    # Predicting Traj2Body
    inputData_ = Variable(torch.from_numpy(t2body_in_std)).cuda()  #(batch, 3, frameNum)
    output = model_traj2body(inputData_)
    tjbody_out_body_np = output.data.cpu().numpy()  #(batch, 73, frames)
    tjbody_out = tjbody_out_body_np[:,:69,:]      #crop the last 4, if there exists
    
    
    tjbody_out = tjbody_out*tj2body_body_std[:,:-4,:] + tj2body_body_mean[:,:-4,:] #Original code
    #tjbody_out = tjbody_out*tj2body_body_std[:,:-4,:]*0.0 + tj2body_body_mean[:,:-4,:] ##Quant: Baseline. Only Mean
    
    ## Optional: Overwrite global trans oreintation info
    tjbody_out[:,-3:,:] =  t2body_in[:,:,:tjbody_out.shape[2]]         

    tjbody_out = np.swapaxes(tjbody_out,1,2)  #(batch, frames, 73)
    tjbody_out = np.reshape(tjbody_out,(-1,69))
    pred_traj2body = np.swapaxes(tjbody_out,0,1)

    """For visualization: Get GT Body with global infomation. Order:Target buyer seller1)"""
    vis_bodyData_GT = [ body_raw_group[2,:,:], body_raw_group[0,:,:], body_raw_group[1,:,:] ]  #(frames, 73)
    for i,X in enumerate(vis_bodyData_GT):
        vis_bodyData_GT[i] =  np.swapaxes(X, 0, 1).astype(np.float32) #(73, frames)

    vis_bodyGT_initTrans = [test_body_initInfo[seqIdx][2]['pos'], test_body_initInfo[seqIdx][0]['pos'], test_body_initInfo[seqIdx][1]['pos'] ]
    vis_bodyGT_initRot = [test_body_initInfo[seqIdx][2]['rot'], test_body_initInfo[seqIdx][0]['rot'], test_body_initInfo[seqIdx][1]['rot']]
    vis_bodyGT_initRot = [ Quaternions(x) for x in vis_bodyGT_initRot ]

    #Prediction, GT, input1, input2
    vis_t2body_bodyData = [ pred_traj2body] + vis_bodyData_GT  
    vis_t2body_initRot = [ pred_initRot_list[0] ] + vis_bodyGT_initRot
    vis_t2body_initTrans = [ pred_initTrans_list[0] ] +  vis_bodyGT_initTrans

    vis_traj_holden = traj_list_holdenForm + gt_traj_list_holdenForm
    vis_traj_holden_initTrans = pred_initTrans_list + gt_initTrans_list
    vis_traj_holden_initRot = pred_initRot_list + gt_initRot_list


    # #######################################################
    # # Quantification: Compute Skeleton Error
    # #Only consider the predicted on
    # HOLDEN_DATA_SCALING = 5
    # bodyData_pred = pred_traj2body[:-3,:]*HOLDEN_DATA_SCALING   #66,frames
    # bodyData_gt = vis_bodyData[0][:-7,:]*HOLDEN_DATA_SCALING       #66, frames
    # bodyData_gt = bodyData_gt[:,:bodyData_pred.shape[1]]
    # skelErr = ( bodyData_pred -  bodyData_gt)**2           #66, frames
    # skelErr = np.reshape(skelErr, (3,22,-1))        #3,22, frames
    # skelErr = np.sqrt(np.sum(skelErr, axis=0))      #22,frames        
    # skelErr = np.mean(skelErr,axis=0)   #frames
    # traj2body_skeletonErr_list.append(skelErr)


    #######################################################
    # Pre-processing for Body2Body
    """Apply Body2Body"""
    ## Test data
    b2b_in_body = body_raw_group[0:1,:,:].copy()      #(1, frames, features:73) //person0,1's all values (position, head orientation, body orientation)
    b2b_in_body = np.concatenate( (b2b_in_body, body_raw_group[1:2,:,:].copy()), axis= 2)      #(1, frames, features:146)
    b2b_ouput_GT = body_raw_group[2:3,:,:].copy()    #(1, frames, features:73)

    # Data pre-processing
    b2b_in_body = np.swapaxes(b2b_in_body, 1, 2).astype(np.float32)   #(1, features, frames)
    b2b_ouput_GT = np.swapaxes(b2b_ouput_GT, 1, 2).astype(np.float32)   #(1, features, frames)

    # Data pre-processing
    b2b_in_body_std = (b2b_in_body - b2b_body_mean_two) / b2b_body_std_two
    b2b_ouput_GT_std = (b2b_ouput_GT - b2b_body_mean) / b2b_body_std

    #######################################################
    # Predicting by Body2Body
    inputData_ = Variable(torch.from_numpy(b2b_in_body_std)).cuda()  #(batch, 3, frameNum)
    output = model_body2body(inputData_)
    output = model_ae_body.decoder(output)


    #De-standardaize
    output_np = output.data.cpu().numpy()  #(batch, 73, frames)
    output_np = output_np*b2b_body_std + b2b_body_mean

    output_np = np.swapaxes(output_np,1,2)  #(batch, frames, 73)
    output_np = np.reshape(output_np,(-1,73))
    b2b_pred_body2body = np.swapaxes(output_np,0,1)  #(73, frames)
    b2b_pred_body2body = b2b_pred_body2body[:69,:]    #(69, frames)

    #Prediction, GT, input1, input2
    vis_b2body_bodyData = [ b2b_pred_body2body] + vis_bodyData_GT  
    vis_b2body_initRot = [ pred_initRot_list[0] ] + vis_bodyGT_initRot
    vis_b2body_initTrans = [ pred_initTrans_list[0] ] +  vis_bodyGT_initTrans


    """Ensure the same length"""
    frameLeng = min(vis_b2body_bodyData[0].shape[1], vis_t2body_bodyData[0].shape[1])
    vis_b2body_bodyData[0] = vis_b2body_bodyData[0][:,:frameLeng]
    vis_t2body_bodyData[0] = vis_t2body_bodyData[0][:,:frameLeng]

    vis_b2body_hybrid_bodyData = [ f.copy() for f in vis_b2body_bodyData ]
    vis_b2body_hybrid_bodyData[0][0:33,:] = vis_t2body_bodyData[0][0:33,:]  #Overwrite leg motion
    vis_b2body_hybrid_bodyData[0][-3:,:] = vis_t2body_bodyData[0][-3:,:]     #Overwrite root motion


    # # ## Compute Skeleton Error
    # #Only consider the predicted on
    # HOLDEN_DATA_SCALING = 5
    # bodyData_pred = pred_body2body[:-3,:]*HOLDEN_DATA_SCALING   #66,frames
    # bodyData_gt = vis_bodyData[0][:-7,:]*HOLDEN_DATA_SCALING       #66, frames
    # bodyData_gt = bodyData_gt[:,:bodyData_pred.shape[1]]
    # skelErr = ( bodyData_pred -  bodyData_gt)**2           #66, frames
    # skelErr = np.reshape(skelErr, (3,22,-1))        #3,22, frames
    # skelErr = np.sqrt(np.sum(skelErr, axis=0))      #22,frames        
    # skelErr = np.mean(skelErr,axis=0)   #frames
    # body2body_skeletonErr_list.append(skelErr)

    #######################################################
    # Pre-processing for Body2Speak
    """Apply Body2Speak"""
    ## Test data
    # input_body = body_raw_group[2:3,:,:].copy()      #Own body
    # ouput_speech_GT = speech_raw[2,:].copy()
    b2speak_in_body = body_raw_group[1:2,:,:].copy()      #other seller's body
    b2speak_out_speech_GT = speech_raw[2,:].copy()
    
    # Data pre-processing
    b2speak_in_body = np.swapaxes(b2speak_in_body, 1, 2).astype(np.float32)   #(1, features, frames)
    b2speak_in_body_std = (b2speak_in_body - b2s_body_mean) / b2s_body_std
    

    #######################################################
    # Predicting by Body2Speak
    inputData_ = Variable(torch.from_numpy(b2speak_in_body_std)).cuda()  #(batch, 3, frameNum)
    output = model_body2speak(inputData_)
    b2speak_pred_speak = output.data.cpu().numpy()  #(batch, 73, frames)  
    b2speak_pred_speak = np.squeeze(b2speak_pred_speak)>0.5
    b2speak_pred_speak = b2speak_pred_speak*1     #0 or 1 value

    # """VIsualize Speacking classification"""
    # import matplotlib.pyplot as plt
    # plt.plot(np.squeeze(speak_output_np))
    # plt.hold(True)
    # plt.plot(ouput_speech_GT)
    # plt.show()

    #Speak, Prediction, GT, Input1, Input2
    vis_b2speak_speak = [b2speak_pred_speak, speech_raw[2,:].copy(), speech_raw[0,:].copy(), speech_raw[1,:].copy()]
    #Speak, GT
    # vis_speech = [speech_raw[2,:].copy(), speech_raw[0,:].copy(), speech_raw[1,:].copy()]


    #######################################################
    # Pre-processing for Face2Face
    #faceData_in = [face_raw_group[0,:,:], face_raw_group[1,:,:] ]  #(frames, 5)
    faceData_in = face_raw_group[0:1,:,:]      #(1, frames, features:5)
    faceData_in = np.concatenate( (faceData_in, face_raw_group[1:2,:,:]), axis= 2)      #(1, frames, features:5
    faceData_in = np.swapaxes(faceData_in, 1, 2).astype(np.float32)   #(1, features:10, frames)

    faceData_in_std = net_face2face.standardize_input(faceData_in)  #(1, features:10, frames)

    #######################################################
    # Predicting face by Face2Face
    inputData_ = Variable(torch.from_numpy(faceData_in_std)).cuda()  #(batch, 3, frameNum)
    f2face_pred_face = net_face2face(inputData_)
    #De-standardaize
    f2face_pred_face = f2face_pred_face.data.cpu().numpy()  #(1, featureDim:5, frames)
    f2face_pred_face = net_face2face.destandardize_output(f2face_pred_face)
 #  output_np = output_np*body_std + body_mean
    f2face_pred_face = np.squeeze(f2face_pred_face)  #(featureDim:5, 73)


    
    vis_faceData_GT = [ face_raw_group[2,:,:], face_raw_group[0,:,:], face_raw_group[1,:,:] ]  #(frames, 5)
    for i,X in enumerate(vis_faceData_GT):
        vis_faceData_GT[i] =  np.swapaxes(X, 0, 1).astype(np.float32) #(5, frames)

    #Order, prediction, gt, intpu1, intput2
    vis_faceData = [f2face_pred_face] +  vis_faceData_GT




    # ####################################################################################
    # # Manually Selecting Data for visualization 
    # """ Select Body to visualize """
    # final_vis_bodyData = [ vis_t2body_bodyData[1], vis_t2body_bodyData[2], vis_t2body_bodyData[3] ]
    # final_vis_body_initRot = [ vis_b2body_initRot[1], vis_b2body_initRot[2], vis_b2body_initRot[3] ] 
    # final_vis_body_initTrans = [ vis_b2body_initTrans[1], vis_b2body_initTrans[2], vis_b2body_initTrans[3] ]

    # """ Select Trajectory to visualize """
    # final_vis_posData = [vis_traj_posData[1], vis_traj_posData[2], vis_traj_posData[3] ]
    # final_vis_faceNormalData = [vis_traj_faceNormalData[1], vis_traj_faceNormalData[2], vis_traj_faceNormalData[3] ]
    # final_vis_bodyNormalData = [vis_traj_bodyNormalData[1], vis_traj_bodyNormalData[2], vis_traj_bodyNormalData[3] ]

    # """ Select HoldenForm-Trajectory to visualize """
    # #Order: (Pred, GT)
    # final_vis_holdenTraj = vis_traj_holden
    # final_vis_holdenTraj_holden_initRot = vis_traj_holden_initRot
    # final_vis_holdenTraj_holden_initTrans = vis_traj_holden_initTrans

    # """ Select Speak to visualize """
    # final_vis_speak = [vis_b2speak_speak[1], vis_b2speak_speak[2], vis_b2speak_speak[3]]

    # """ Select Face to visualize """
    # #Final slection to draw a subset
    # final_vis_faceData = [ vis_faceData[0], vis_faceData[2], vis_faceData[3] ]

    ####################################################################################
    # Manually Selecting Data for visualization 
    selectedIdx = [0, 1, 2, 3]
    """ Select Body to visualize """
    final_vis_body_initRot = [ vis_b2body_initRot[i] for i in selectedIdx]
    final_vis_body_initTrans = [ vis_b2body_initTrans[i] for i in selectedIdx]
    #final_vis_bodyData = [ vis_t2body_bodyData[i] for i in selectedIdx]        #Traj2Body
    final_vis_bodyData = [ vis_b2body_hybrid_bodyData[i] for i in selectedIdx]        #Hybrid
    #final_vis_bodyData = [vis_t2body_bodyData[0], vis_b2body_bodyData[0], vis_b2body_hybrid_bodyData[0], vis_b2body_bodyData[1]]
    #final_vis_bodyData = [vis_t2body_bodyData[0], vis_b2body_hybrid_bodyData[0], vis_b2body_bodyData[1]]
    

    """ Select Trajectory to visualize """
    final_vis_posData = [vis_traj_posData[i] for i in selectedIdx]
    final_vis_faceNormalData = [vis_traj_faceNormalData[i] for i in selectedIdx]
    final_vis_bodyNormalData = [vis_traj_bodyNormalData[i] for i in selectedIdx]

    # """ Select HoldenForm-Trajectory to visualize """
    # #Order: (Pred, GT)
    final_vis_holdenTraj = vis_traj_holden
    final_vis_holdenTraj_holden_initRot = vis_traj_holden_initRot
    final_vis_holdenTraj_holden_initTrans = vis_traj_holden_initTrans

    """ Select Speak to visualize """
    final_vis_speak = [vis_b2speak_speak[i] for i in selectedIdx]

    """ Select Face to visualize """
    #Final slection to draw a subset
    final_vis_faceData = [ vis_faceData[i] for i in selectedIdx]

    if bVisualize==False:
        continue

    ####################################################################################
    # Visualization
    """ Visualize Location + Orientation """
    glViewer.setPosOnly(final_vis_posData)
    glViewer.setFaceNormal(final_vis_faceNormalData)
    glViewer.setBodyNormal(final_vis_bodyNormalData)
    
    """Visualize Trajectory"""
    #glViewer.set_Holden_Trajectory_3(final_vis_holdenTraj, initTrans=final_vis_holdenTraj_holden_initTrans, initRot=final_vis_holdenTraj_holden_initRot)   #pred

    """Visualize Body"""
    glViewer.set_Holden_Data_73(final_vis_bodyData, ignore_root=False, initRot=final_vis_body_initRot, initTrans= final_vis_body_initTrans, bIsGT=False)
    
    """Visualize Speech"""
    glViewer.setSpeech_binary(final_vis_speak)

    """Visualize face"""
    #glViewer.SetFaceParmData(vis_faceData,False)
    vis_f2face_trans , vis_f2face_rot = utility.ComputeHeadOrientation(glViewer.g_skeletons, final_vis_faceNormalData)
    glViewer.SetFaceParmDataWithTrans(final_vis_faceData,True, vis_f2face_trans, vis_f2face_rot)

    """Render output to videos"""
    if bRender:
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




