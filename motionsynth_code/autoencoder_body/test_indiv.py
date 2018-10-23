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

# Utility Functions
from utility import print_options,save_options
from utility import setCheckPointFolder
from utility import my_args_parser


#by jhugestar
sys.path.append('/ssd/codes/glvis_python/')
#from glViewer import SetFaceParmData,setSpeech,setSpeechGT,setSpeech_binary, setSpeechGT_binary, init_gl #opengl visualization 
import glViewer



######################################3
# Logging
import logging
#FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
FORMAT = '[%(levelname)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)  ##default logger


######################################3
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
test_dblist = ['data_hagglingSellers_speech_body_120frm_10gap_white_testing']

test_data = np.load(datapath + test_dblist[0] + '.npz')
test_X_raw = test_data['clips']  #Input (1044,240,73)
test_speech_raw  = test_data['speech']  #Input (1044,240,73)

# test_X = test_X[:100,:,:]
# test_Y = test_Y[:100]

######################################
# Checkout Folder and pretrain file setting
checkpointRoot = './'
checkpointFolder = checkpointRoot+ '/social_autoencoder_3conv_vect_vae_try7/'
preTrainFileName= 'checkpoint_e22_loss0.0627.pth'


######################################
# Input/Output Option
test_X = test_X_raw#[1,:,:,:]      #(num, chunkLength, dim:200) //person 1 (first seller's value)
#test_Y = test_X_raw[2,:,:,:]

######################################
# Data pre-processing
test_X = np.swapaxes(test_X, 1, 2).astype(np.float32) #(num, frames, dim:200) => (num, 200, frames)
#test_Y = np.swapaxes(test_Y, 1, 2).astype(np.float32) #(num, frames, dim:200) => (num, 200, frames)
#train_Y = train_Y.astype(np.float32)



######################################
# Data pre-processing
preprocess = np.load(checkpointFolder + 'preprocess_core.npz') #preprocess['Xmean' or 'Xstd']: (1, 73,1)

Xmean = preprocess['Xmean']
Xstd = preprocess['Xstd']

#Ymean = preprocess['Ymean']
#Ystd = preprocess['Ystd']

test_X_std = (test_X - Xmean) / Xstd
#test_Y_std = (test_Y - Xmean) / Xstd


######################################
# Network Setting
#batch_size = args.batch
batch_size = 1#512
#criterion = nn.BCELoss()
criterion = nn.MSELoss()

#Creat Model
#model = modelZoo.autoencoder_first().cuda()
featureDim = test_X_raw.shape[2]
latentDim = 200
model = modelZoo.autoencoder_3conv_vect_vae(featureDim,latentDim).cuda()
model.eval()

#Creat Model
trainResultName = checkpointFolder + preTrainFileName
loaded_state = torch.load(trainResultName, map_location=lambda storage, loc: storage)

model.load_state_dict(loaded_state,strict=False) #strict False is needed some version issue for batchnorm
model = model.eval()  #Do I need this again?



# Ystd = np.swapaxes(Ystd,1,2)
# Ymean = np.swapaxes(Ymean,1,2)

######################################
# Training
batchinds = np.arange(test_X.shape[0] // batch_size)
pred_all = np.empty([0,1],dtype=float)
test_X_vis =np.empty([0,200],dtype=float)

featureDim = 10
for _, bi in enumerate(batchinds):

    if bi %10 !=0:
        continue

    idxStart  = bi*batch_size
    inputData_np = test_X_std[idxStart:(idxStart+batch_size),:,:]
    inputData_np_ori = test_X[idxStart:(idxStart+batch_size),:,:]
    # outputData_np = test_Y_std[idxStart:(idxStart+batch_size),:,:]
    # outputData_np_ori = test_Y[idxStart:(idxStart+batch_size),:,:]

    inputData = Variable(torch.from_numpy(inputData_np)).cuda()  #(batch, 73, frameNum)
    # outputGT = Variable(torch.from_numpy(outputData_np)).cuda()  #(batch, 73, frameNum)
    #outputGT = Variable(torch.from_numpy(inputData_np)).cuda()  #(batch, 73, frameNum)

    # ===================forward=====================
    output, mu, logvar = model(inputData)
    #loss = criterion(output, outputGT)
    loss = criterion(output, inputData)


    #De-standardaize
    output_np = output.data.cpu().numpy()  #(batch, 73, frames)
    output_np = output_np*Xstd + Xmean

    output_np = np.swapaxes(output_np,1,2)  #(batch, frames, 73)
    output_np = np.reshape(output_np,(-1,73))
    output_np = np.swapaxes(output_np,0,1)



    inputData_np_ori = np.swapaxes(inputData_np_ori,1,2)  #(batch, frames, 73)
    inputData_np_ori = np.reshape(inputData_np_ori,(-1,73))
    inputData_np_ori = np.swapaxes(inputData_np_ori,0,1)

    # outputData_np_ori = np.swapaxes(outputData_np_ori,1,2)  #(batch, frames, 73)
    # outputData_np_ori = np.reshape(outputData_np_ori,(-1,73))
    # outputData_np_ori = np.swapaxes(outputData_np_ori,0,1)

    #glViewer.show_Holden_Data_73([ outputData_np_ori, inputData_np_ori, output_np] )
    glViewer.set_Holden_Data_73([inputData_np_ori, output_np] )
    glViewer.init_gl()


    continue

    #Save Speaking prediction
    pred = output.data.cpu().numpy()
    pred_all = np.concatenate((pred_all, pred[:,-1]), axis=0)

    test_X_vis = np.concatenate((test_X_vis, inputData_np_ori), axis=0)


#Computing Accuracy
pred_binary = pred_all[:] >=0.5
pred_binary = pred_binary[:,-1]
from sklearn.metrics import accuracy_score
test_Y_cropped = test_Y[:len(pred_binary),-1]
#acc = accuracy_score(test_Y_, pred_binary)

t = (test_Y_cropped == pred_binary)
correct_samples = sum(t)
acc = float(correct_samples)/len(test_Y_cropped)
print('Testing accuracy: {0:.2f}% (={1}/{2})'.format(acc*100.0,correct_samples,len(test_Y_cropped)))


bPlot = False
if bPlot:
    # add a subplot with no frame
    import matplotlib.pyplot as plt
    plt.subplot(221)
    ax2=plt.subplot(311)
    plt.plot(test_Y)
    plt.title('Speech GT')
    ax2=plt.subplot(312)
    plt.plot(pred_binary)
    plt.title('Prediction (binary)')
    ax2=plt.subplot(313)
    plt.plot(pred_all)
    plt.title('Prediction (probability)')
    #plt.ion()
    plt.show()
    #plt.pause(1)

bVisualize = False
if bVisualize:
    #by jhugestar
    sys.path.append('/ssd/codes/glvis_python/')
    #from glViewer import SetFaceParmData,setSpeech,setSpeechGT,setSpeech_binary, setSpeechGT_binary, init_gl #opengl visualization 
    import glViewer


    maxFrameNum = 2000
    frameNum = test_X_vis.shape[0]
    startIdx = 0
    test_X_vis = np.swapaxes(test_X_vis, 0, 1) #(frames, 200) ->(200, frames) where num can be thought as frames
    
    while startIdx< frameNum:
            
        endIdx = min(frameNum, startIdx + maxFrameNum)

        glViewer.setSpeechGT_binary([test_Y_cropped[startIdx:endIdx]])
        glViewer.setSpeech_binary([pred_binary[startIdx:endIdx]])


        glViewer.SetFaceParmData([test_X_vis[:, startIdx:endIdx]])
        glViewer.init_gl()

        startIdx += maxFrameNum
