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

#import modelZoo

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


test_dblist_body = ['data_hagglingSellers_speech_body_face_bySequence_white_noGa_brl_testing']
test_dblist_body = ['data_hagglingSellers_speech_body_face_bySequence_white_noGa_brl_testing_tiny']


"""Load body motion data"""
pkl_file_body = open(datapath + test_dblist_body[0] + '.pkl', 'rb')
test_data_body = pickle.load(pkl_file_body)
pkl_file_body.close()

test_body_raw_all = test_data_body['data']  #Input (1044,240,73)
test_body_seqNames = test_data_body['seqNames']
test_body_initInfo = test_data_body['initInfo']

"""Load face expression data"""
test_face_raw_all = test_data_body['face']  #Input (1044,240,73)

# ######################################
# # Load Face2body
from network_face2body import Network_face2body 
net_face2body = Network_face2body()


bVisualize = True


for seqIdx in range(len(test_face_raw_all)):

    # if seqIdx!=len(test_X_raw_all)-1:
    #     continue

    #print('{}'.format(os.path.basename(test_seqNames[seqIdx])))

    face_raw_group = test_face_raw_all[seqIdx]     #(1, frames, feature:73)
    test_body_group = test_body_raw_all[seqIdx]     #(1, frames, feature:73)

    #######################################################
    # Pre-processing for Face2Face
    #faceData_in = [face_raw_group[0,:,:], face_raw_group[1,:,:] ]  #(frames, 5)
    faceData_in = face_raw_group[2:3,:,:]      #(1, frames, features:5)
    faceData_in = np.swapaxes(faceData_in, 1, 2).astype(np.float32)   #(1, features:10, frames)

    faceData_in_std = net_face2body.standardize_input(faceData_in)  #(1, features:10, frames)

    #######################################################
    # Predicting face by Face2Face
    inputData_ = Variable(torch.from_numpy(faceData_in_std)).cuda()  #(batch, 3, frameNum)
    f2face_pred_body = net_face2body(inputData_)
    #De-standardaize
    f2face_pred_body = f2face_pred_body.data.cpu().numpy()  #(1, featureDim:5, frames)
    f2face_pred_body = net_face2body.destandardize_output(f2face_pred_body)
 #  output_np = output_np*body_std + body_mean
    f2face_pred_body = np.squeeze(f2face_pred_body)  #(featureDim:73, frames)

    bodyData = [ f2face_pred_body]
    glViewer.set_Holden_Data_73(bodyData)#, initTrans=initTrans, initRot=initRot)
    glViewer.init_gl()

    continue

    # #Output GT
    # outputData_np_GT = np.swapaxes(outputData_np_GT,1,2)  #(batch, frames, 73)
    # outputData_np_GT = np.reshape(outputData_np_GT,(-1,73))
    # outputData_np_GT = np.swapaxes(outputData_np_GT,0,1)
  
    # #Input GTs
    # inputData_np_ori_1 = inputData_np_ori[:,:73,:]
    # inputData_np_ori_1 = np.swapaxes(inputData_np_ori_1,1,2)  #(batch, frames, 73)
    # inputData_np_ori_1 = np.reshape(inputData_np_ori_1,(-1,73))
    # inputData_np_ori_1 = np.swapaxes(inputData_np_ori_1,0,1)

    # inputData_np_ori_2 = inputData_np_ori[:,73:,:]
    # inputData_np_ori_2 = np.swapaxes(inputData_np_ori_2,1,2)  #(batch, frames, 73)
    # inputData_np_ori_2 = np.reshape(inputData_np_ori_2,(-1,73))
    # inputData_np_ori_2 = np.swapaxes(inputData_np_ori_2,0,1)

    # #glViewer.show_Holden_Data_73([ outputData_np_ori, inputData_np_ori, output_np] )

    # initTrans = [outputData_initTrans,outputData_initTrans, inputData_initTrans, inputData_initTrans2]
    # initRot = [outputData_initRot[0],outputData_initRot[0], inputData_initRot[0], inputData_initRot2[0]]
    # frameLen = output_np.shape[1]
    


    # """Remvoe OUTPU GT"""
    # bodyData = [bodyData[0],bodyData[2],bodyData[3]]
    # initRot = [initRot[0],initRot[2],initRot[3]]
    # initTrans = [initTrans[0],initTrans[2],initTrans[3]]



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
    
    # if bVisualize==False:
    #     continue

    # glViewer.set_Holden_Data_73(bodyData, initTrans=initTrans, initRot=initRot)#, initTrans=initTrans, initRot=initRot)
    # glViewer.init_gl()

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




