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


test_dblist = ['data_hagglingSellers_speech_body_bySequence_white_noGa_testing_tiny']

pkl_file = open(datapath + test_dblist[0] + '.pkl', 'rb')
test_data = pickle.load(pkl_file)
pkl_file.close()

test_X_raw_all = test_data['data']  #Input (1044,240,73)
# test_Y_raw_all = test_data['speech']  #Input (1044,240,73)
test_seqNames = test_data['seqNames']

######################################
# Checkout Folder and pretrain file setting

checkpointRoot = './'
checkpointFolder = checkpointRoot+ '/social_regressor_holden_73_try4/'
preTrainFileName= 'checkpoint_e5300_loss0.2356.pth'


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


######################################
# If model is for single frame data
bSingleFrameInput = False
if not hasattr(model, 'init_hidden'):
    bSingleFrameInput = True

############################
## Choose a sequence
#seqIdx =1

posErr_list = []

bVisualize = True
for seqIdx in range(len(test_X_raw_all)):

    # if seqIdx!=len(test_X_raw_all)-1:
    #     continue

    print('{}'.format(os.path.basename(test_seqNames[seqIdx])))

    test_X_raw = test_X_raw_all[seqIdx][1:2,:,:]     #(1, frames, feature:3)

    ######################################
    # Input/Output Option
    test_traj = test_X_raw[:,:,-7:-4].copy()    #(1, frameNum, feature:3)
    test_body = test_X_raw

    ######################################
    # Data pre-processing
    test_traj = np.swapaxes(test_traj, 1, 2).astype(np.float32) #(num, 3, frameNum)
    test_body = np.swapaxes(test_body, 1, 2).astype(np.float32) #(num, 73, frameNum)


    ######################################
    # Data pre-processing
    preprocess = np.load(checkpointFolder + 'preprocess_core.npz') #preprocess['Xmean' or 'Xstd']: (1, 73,1)

    body_mean = preprocess['body_mean']
    body_std = preprocess['body_std']

    traj_mean = preprocess['traj_mean']
    traj_std = preprocess['traj_std']

    test_body_std = (test_body - body_mean) / body_std
    test_traj_std = (test_traj - traj_mean) / traj_std



    idxStart  = 0 
    batch_size = test_traj_std.shape[0]
    inputData_np = test_traj_std[idxStart:(idxStart+batch_size),:,:]  #(batch, 3, frameNum)
    outputData_np = test_body_std[idxStart:(idxStart+batch_size),:-4,:] #(batch, 73 - 4, frameNum)
    
    inputData_np_ori = test_traj[idxStart:(idxStart+batch_size),:,:]  #(batch, 3, frameNum)
    outputData_np_ori = test_body[idxStart:(idxStart+batch_size),:-4,:] #(batch, 73 - 4, frameNum)

    inputData = Variable(torch.from_numpy(inputData_np)).cuda()  #(batch, 3, frameNum)
    outputGT = Variable(torch.from_numpy(outputData_np)).cuda()  #(batch, 73, frameNum)

    # ===================forward=====================
    output = model(inputData)
    #loss = criterion(output, outputGT)
    #loss = criterion(output, outputGT)


    #De-standardaize
    output_np = output.data.cpu().numpy()  #(batch, 73, frames)
    output_np = output_np[:,:69,:]      #crop the last 4, if there exists
    output_np = output_np*body_std[:,:-4,:] + body_mean[:,:-4,:]
    #output_np[:,-3:,:] =  inputData_np_ori         #Overwrite global trans oreintation info

    output_np = np.swapaxes(output_np,1,2)  #(batch, frames, 73)
    output_np = np.reshape(output_np,(-1,69))
    output_np = np.swapaxes(output_np,0,1)
  
  
    

    inputData_np_ori = np.swapaxes(inputData_np_ori,1,2)  #(batch, frames, 73)
    inputData_np_ori = np.reshape(inputData_np_ori,(-1,3))
    inputData_np_ori = np.swapaxes(inputData_np_ori,0,1)

    outputData_np_ori = np.swapaxes(outputData_np_ori,1,2)  #(batch, frames, 73)
    outputData_np_ori = np.reshape(outputData_np_ori,(-1,69))
    outputData_np_ori = np.swapaxes(outputData_np_ori,0,1)

    #glViewer.show_Holden_Data_73([ outputData_np_ori, inputData_np_ori, output_np] )

    if bVisualize==False:
        continue

    frameLength = output_np.shape[1]
    glViewer.set_Holden_Data_73([output_np[:,:frameLength], outputData_np_ori[:,:frameLength]] )
    glViewer.set_Holden_Trajectory_3([inputData_np_ori[:,:frameLength], inputData_np_ori[:,:frameLength]] )
    glViewer.init_gl()





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


# add a subplot with no frame
bShowGraph = False
if bShowGraph:
    import matplotlib.pyplot as plt
    ax2=plt.subplot(111)
    plt.plot(avg_posErr_list)
    plt.title('Average Pos Error')
    plt.show()

