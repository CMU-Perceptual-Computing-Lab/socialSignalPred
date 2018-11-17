"""
Input: Positions from two subjects (4dim)
Output: Position + FaceRotations (4dim)
"""

import os
import sys
import numpy as np
import scipy.io as io
import random

######################################3
# Logging
import logging
#FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
FORMAT = '[%(levelname)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)  ##default logger


# Tensorboard logging. 
tensorboard_bLog = False  
try:
	sys.path.append('../utils')
	from logger import Logger
	tensorboard_bLog = True
except ImportError:
	pass

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

######################################
# Parameter Handling
parser = my_args_parser()
args = parser.parse_args()


######################################
# Manual Setting
args.model = 'regressor_fcn'
#args.model = 'regressor_fcn_3'

# Some initializations #
torch.cuda.set_device(args.gpu)

rng = np.random.RandomState(23456)
torch.manual_seed(23456)
torch.cuda.manual_seed(23456)


######################################
# Dataset 
#datapath ='/ssd/codes/pytorch_motionSynth/motionsynth_data' 
bBRLSorting = False
datapath ='../../motionsynth_data/data/processed/' 

# #Pos -> Pos only
# train_dblist = ['data_hagglingSellers_speech_formation_pN_rN_rV_240frm_5gap_white_training']
# test_dblist = ['data_hagglingSellers_speech_formation_pN_rN_rV_240frm_5gap_white_testing']

#BRL sorting
# train_dblist = ['data_hagglingSellers_speech_formation_pN_rN_rV_240frm_5gap_white_brl_training']
# test_dblist = ['data_hagglingSellers_speech_formation_pN_rN_rV_240frm_5gap_white_brl_testing']


#BRL, no normalization_tiny
train_dblist = ['data_hagglingSellers_speech_formation_240frm_5gap_white_brl_testing_tiny']
test_dblist = ['data_hagglingSellers_speech_formation_240frm_5gap_white_brl_testing_tiny']       


#BRL, normalization by the first frame (pos only)
train_dblist = ['data_hagglingSellers_speech_formation_240frm_5gap_white_brl_firstFrmPosNorm_training']
test_dblist = ['data_hagglingSellers_speech_formation_240frm_5gap_white_brl_firstFrmPosNorm_testing']  

#BRL, no normalization
train_dblist = ['data_hagglingSellers_speech_formation_240frm_5gap_white_brl_training']
test_dblist = ['data_hagglingSellers_speech_formation_240frm_5gap_white_brl_testing']       


bBRLSorting = True

train_data = np.load(datapath + train_dblist[0] + '.npz')

train_X_raw= train_data['clips']  #Input (3, numClip, chunkLengh, 9)  where 9 dim represents [ pos;faceNormal;bodyNormal ]
train_Y_raw = train_data['speech']  #Input (3, numClip, chunkLengh)

test_data = np.load(datapath + test_dblist[0] + '.npz')
test_X_raw= test_data['clips']      #Input (3, numClip, chunkLengh, 9)  where 9 dim represents [ pos;faceNormal;bodyNormal ]
test_Y_raw = test_data['speech']    #Input (3, numClip, chunkLengh)


######################################
# Input/Output Option


# Feature Selection
#features = (0,2, 3,5, 6,8)      #if train_X_raw.shape[-1] ==9:
#features = (0,2, 3,4)           #if train_X_raw.shape[-1] ==4:
#input_features = (0,2, 3,5 , 9, 11, 12, 14) #Position Only -> Position
input_features = (0,2, 3,5,6,8,  9,11, 12,14, 15,17) #Position + FaceOri
#output_features = (0,2, 3,5) #Position + face normal
output_features = (0,2, 3,5, 6,8) #Position + face normal

#train_X_raw = train_X_raw[:,:,:,features]
#test_X_raw = test_X_raw[:,:,:,features]

train_X = train_X_raw[0,:,:,:]      #(num, chunkLength, 9) //person0,1's all values (position, head orientation, body orientation)
train_X = np.concatenate( (train_X, train_X_raw[1,:,:,:]), axis= 2)    #(num, chunkLength, 18)
train_X = train_X[:,:,input_features]
train_Y = train_X_raw[2,:,:,:]    #2nd seller's position only
train_Y = train_Y[:,:,output_features]

## Change the second and third 
if bBRLSorting== False:
    train_X_swap = train_X_raw[0,:,:,:]      #(num, chunkLength, 9) //person0,1's all values (position, head orientation, body orientation)
    train_X_swap = np.concatenate( (train_X_swap, train_X_raw[2,:,:,:]), axis= 2)    #(num, chunkLength, 18)
    train_X_swap = train_X_swap[:,:,input_features]
    train_Y_swap = train_X_raw[1,:,:,:]    #2nd seller's position only
    train_Y_swap = train_Y_swap[:,:,output_features]

    train_X = np.concatenate( (train_X, train_X_swap), axis=0)
    train_Y = np.concatenate( (train_Y, train_Y_swap), axis=0)


## Test data
test_X = test_X_raw[0,:,:,:]      #(num, chunkLength, 9) //person0,1's all values (position, head orientation, body orientation)
test_X = np.concatenate( (test_X, test_X_raw[1,:,:,:]), axis= 2)      #(num, chunkLength, 18)
test_X = test_X[:,:,input_features]
test_Y = test_X_raw[2,:,:,:]    #2nd seller's position only
test_Y = test_Y[:,:,output_features]


if bBRLSorting== False:
    test_X_swap = test_X_raw[0,:,:,:]      #(num, chunkLength, 9) //person0,1's all values (position, head orientation, body orientation)
    test_X_swap = np.concatenate( (test_X_swap, test_X_raw[2,:,:,:]), axis= 2)      #(num, chunkLength, 18)
    test_X_swap = test_X_swap[:,:,input_features]
    test_Y_swap = test_X_raw[1,:,:,:]    #2nd seller's position only
    test_Y_swap = test_Y_swap[:,:,output_features]

    test_X = np.concatenate( (test_X, test_X_swap), axis=0)
    test_Y = np.concatenate( (test_Y, test_Y_swap), axis=0)



# train_X = train_X[:-1:10,:,:]
# train_Y = train_Y[:-1:10,:]

######################################
# Network Setting
num_epochs = args.epochs #500
#batch_size = 128
batch_size = args.batch
learning_rate = 1e-3

# if args.autoreg ==1: #and "vae" in args.model:
#     model = getattr(modelZoo,args.model)(frameLeng=160).cuda()
# else:
#     model = getattr(modelZoo,args.model)().cuda()

#model = modelZoo.naive_lstm(batch_size).cuda()
#model = modelZoo.naive_lstm_embBig(batch_size).cuda()
#model = modelZoo.naive_mlp().cuda()
#model = modelZoo.naive_mlp_2().cuda()
#model = modelZoo.naive_mlp_wNorm_2().cuda()
model = getattr(modelZoo,args.model)().cuda()
model.train()

for param in model.parameters():
    print(type(param.data), param.size())

# Loss Function #
#criterion = nn.BCELoss()
criterion = nn.MSELoss()

# Solver #
if args.solver == 'adam':
    logger.info('solver: Adam')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
elif args.solver == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    logger.info('solver: SGD')
elif args.solver == 'adam_ams': #only for pytorch 0.4 or later.
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5, amsgrad=True)
    logger.info('solver: Adam with AMSGrad')
else:
    logger.info('Unknown solver option')
    assert(False)


######################################
# Set Check point folder
checkpointFolder = setCheckPointFolder(args, model)
checkpointFolder_base = os.path.basename(checkpointFolder) 


######################################
# Load pre-trained parameters
pretrain_epoch = 0
pretrain_batch_size =args.batch  #Assume current batch was used in pretraining
if args.finetune != '':
    from utility import loadPreTrained
    model, optimizer, pretrain_epoch, pretrain_batch_size = loadPreTrained(args, checkpointFolder, model, optimizer)


######################################
# Log file path + Tensorboard Logging
fileHandler = logging.FileHandler(checkpointFolder+'/train_log.txt')  #to File
logger.addHandler(fileHandler)
if tensorboard_bLog:
    tb_logger = Logger(checkpointFolder+'/logs')  #tensorboard logger

# Save Option Info 
option_str, options_dict = print_options(parser,args)
save_options(checkpointFolder, option_str, options_dict)


######################################
# Data pre-processing
train_X = np.swapaxes(train_X, 1, 2).astype(np.float32) #(num, chunkLength, featureDim) ->(num, featureDim, chunkLength)
train_Y = np.swapaxes(train_Y, 1, 2).astype(np.float32) #(num, chunkLength, featureDim) ->(num, featureDim, chunkLength)

test_X = np.swapaxes(test_X, 1, 2).astype(np.float32) 
test_Y = np.swapaxes(test_Y, 1, 2).astype(np.float32) 

# Compute mean and var
Xmean = train_X.mean(axis=2).mean(axis=0)[np.newaxis,:,np.newaxis]
#Xstd = np.array([[[train_X.std()]]]).repeat(train_X.shape[1], axis=1)          #Bug...std are from all channels..
Xstd = train_X.std(axis=2).std(axis=0)[np.newaxis,:,np.newaxis]

#The following channels are always zero
# Xmean[0,0:1,0] = 0

# Xmean[0,4,0] = 0        #body orientation
# Xmean[0,5,0] = 1        #body orientation

# Xstd[0,(0,1),0] = 1      #avoid dividing by zero for (0,1)
# Xstd[0,(4,5),0] = 1        #avoid dividing by zero

# Data standardization 
train_X = (train_X - Xmean) / Xstd
test_X = (test_X - Xmean) / Xstd

#Standardize output data as well
Ymean = train_Y.mean(axis=2).mean(axis=0)[np.newaxis,:, np.newaxis]
#Ystd = np.array([[[train_Y.std()]]]).repeat(train_Y.shape[1], axis=1)      #bug...
Ystd = train_Y.std(axis=2).std(axis=0)[np.newaxis,:,np.newaxis] 

train_Y = (train_Y - Ymean) / Ystd
test_Y = (test_Y - Ymean) / Ystd

# Save mean and var
np.savez_compressed(checkpointFolder+'/preprocess_core.npz', Xmean=Xmean, Xstd=Xstd, Ymean=Ymean, Ystd=Ystd)

# Data Shuffle
I = np.arange(len(train_X))
rng.shuffle(I); 
train_X = train_X[I]
train_Y = train_Y[I]


######################################################vv
# For Ablation Study
# Data Masking. train_X: (num, feature:12 (pos, face, normal, frames:240)
"""Pos only"""
# mask = (2,3,4,5, 8,9,10,11)
# train_X[:,mask,:] = Xmean[:, mask,:]

"""Pos + face ori only"""
# mask = (4,5, 10,11)
# train_X[:,mask,:] = Xmean[:, mask,:]


"""Pos + body ori only"""
mask = (2,3,  8,9)
train_X[:,mask,:] = Xmean[:, mask,:]


logger.info('Input data size: {0}'.format(train_X.shape))


######################################
# Some settings before training
if train_X.shape[0]  < batch_size:
    batch_size = train_X.shape[0]
curBestAccu = 1e5
#Compute stepNum start point (to be continuos in tensorboard if pretraine data is loaded)
filelog_str = ''
stepNum = pretrain_epoch* len(np.arange(train_X.shape[0] // pretrain_batch_size))



######################################
# Training
bAddNoise = False

for epoch in range(num_epochs):
    
    model.train()

    batchinds = np.arange(train_X.shape[0] // batch_size)
    rng.shuffle(batchinds)

    # Each Batch
    avgLoss =0
    acc = 0
    cnt = 0
    for bii, bi in enumerate(batchinds):

        idxStart  = bi*batch_size

        inputData_np = train_X[idxStart:(idxStart+batch_size),:,:]          #(batchsize,featureDim,frames)
        outputData_np = train_Y[idxStart:(idxStart+batch_size),:]   #(batch, frame, featureDim)
        
        #numpy to tensor
        if bAddNoise:
            #Add noise
            inputData = Variable(torch.from_numpy(inputData_np))
            noise = torch.randn_like(inputData)
            inputData = inputData + noise #* 0.01
            inputData = inputData.cuda()
        else:
            inputData = Variable(torch.from_numpy(inputData_np)).cuda()
        outputGT = Variable(torch.from_numpy(outputData_np)).cuda()
    

        # ===================forward=====================
        output = model(inputData) #output: (batch, frames, inputDim(9))


        l1_reg = None
        for W in model.parameters():
            if l1_reg is None:
                l1_reg = W.norm(1)
            else:
                l1_reg = l1_reg + W.norm(1)        
        l1_regularization = 0.1 * l1_reg

        loss = criterion(output, outputGT) + l1_regularization


        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================log========================
        avgLoss += loss.item()*batch_size

        # compute accuracy
        #correct = (outputGT[:,-1].eq( (output[:,-1]>0.5).float() )).sum() #Just check the last one
        #acc += correct.item()
        #cnt += batch_size

        
        if tensorboard_bLog:
            # 1. Log scalar values (scalar summary)
            if int(torch.__version__[2])==2:
                info = { 'loss': loss.data[0] }
            else:
                info = { 'loss': loss.item() }

            for tag, value in info.items():
                tb_logger.scalar_summary(tag, value, stepNum)
        
        stepNum = stepNum+1

    ######################################
    # Logging
    #temp_str = 'model: {}, epoch [{}/{}], avg loss:{:.4f}, accuracy:{:.4f}'.format(checkpointFolder_base, epoch +pretrain_epoch, num_epochs, avgLoss/ (len(batchinds)*batch_size), float(acc)/cnt )
    temp_str = 'model: {}, epoch [{}/{}], avg loss:{:.4f}'.format(checkpointFolder_base, epoch +pretrain_epoch, num_epochs, avgLoss/ (len(batchinds)*batch_size))
    logger.info(temp_str)


    ######################################
    # Check Testing Error
    batch_size_test = batch_size
    test_loss = 0
    acc= 0.0
    cnt =0.0

    model.eval()
    batchinds = np.arange(test_X.shape[0] // batch_size_test)
    for bii, bi in enumerate(batchinds):

        idxStart  = bi*batch_size_test

        inputData_np = test_X[idxStart:(idxStart+batch_size),:,:]      
        outputData_np = test_Y[idxStart:(idxStart+batch_size),:]   #(batch, frame, featureDim)
        
        #numpy to Tensors
        inputData = Variable(torch.from_numpy(inputData_np)).cuda()
        outputGT = Variable(torch.from_numpy(outputData_np)).cuda()
    

        outputData = model(inputData) #output: (batch, frames, 1)
        loss = criterion(outputData, outputGT)

        test_loss += loss.item()*batch_size_test
        #test_loss += loss.data.cpu().numpy().item()* batch_size_test # sum up batch loss

        # compute accuracy
        #correct = (outputGT[:,-1].eq( (output[:,-1]>0.5).float() )).sum() #Just check the last one
        #acc += correct.item()
        #cnt += batch_size


    test_loss /= len(batchinds)*batch_size_test
    #newAcc = float(acc)/cnt
    
    logger.info('    On testing data: average loss: {:.4f} (best {:.4f})\n'.format(test_loss, curBestAccu))
    if tensorboard_bLog:
        info = { 'test_loss': test_loss }
        for tag, value in info.items():
            tb_logger.scalar_summary(tag, value, stepNum)
        
    bNewBest = False
    if curBestAccu > test_loss:
        curBestAccu = test_loss
        bNewBest = True

    ######################################
    # Save parameters, if current results are the best
    if bNewBest or (epoch + pretrain_epoch) % args.checkpoint_freq ==0 or ( (epoch + pretrain_epoch<50) and ((epoch + pretrain_epoch)%10 ==0) ):
    #if (epoch + pretrain_epoch) % args.checkpoint_freq == 0:
    #if (epoch + pretrain_epoch) % 1 == 0:
        fileName = checkpointFolder+ '/checkpoint_e' + str(epoch + pretrain_epoch) + '_loss{:.4f}'.format(test_loss) + '.pth'
        torch.save(model.state_dict(), fileName)
        fileName = checkpointFolder+ '/opt_state.pth'    #overwrite
        torch.save(optimizer.state_dict(), fileName)
        #torch.save(model, fileName)
