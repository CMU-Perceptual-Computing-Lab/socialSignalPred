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

import cPickle as pickle

import modelZoo

# Utility Functions
from utility import print_options,save_options
from utility import setCheckPointFolder
from utility import my_args_parser

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

train_dblist = ['data_hagglingSellers_speech_body_group_120frm_30gap_white_noGa_training']
test_dblist = ['data_hagglingSellers_speech_body_group_120frm_30gap_white_noGa_testing']

train_dblist = ['data_hagglingSellers_speech_body_group_120frm_30gap_white_noGa_testing_tiny']
test_dblist = ['data_hagglingSellers_speech_body_group_120frm_30gap_white_noGa_testing_tiny']

train_dblist = ['data_hagglingSellers_speech_body_group_120frm_10gap_white_noGa_training']
test_dblist = ['data_hagglingSellers_speech_body_group_120frm_30gap_white_noGa_testing']

# train_dblist = ['data_hagglingSellers_speech_body_face_group_120frm_10gap_white_noGa_testing_tiny']
# test_dblist = ['data_hagglingSellers_speech_body_face_group_120frm_10gap_white_noGa_testing_tiny']

#White Body
train_dblist = ['data_hagglingSellers_speech_body_face_group_120frm_10gap_white_noGa_training']
test_dblist = ['data_hagglingSellers_speech_body_face_group_120frm_10gap_white_noGa_testing']

#White Face&Body
train_dblist = ['data_hagglingSellers_speech_body_face_group_120frm_10gap_whiteBF_noGa_training']
test_dblist = ['data_hagglingSellers_speech_body_face_group_120frm_10gap_whiteBF_noGa_testing']

train_data = np.load(datapath + train_dblist[0] + '.npz')
train_body_raw= train_data['body']  #Input (3, num ,240,73)
train_face_raw= train_data['face']  #Input (3, num ,240,73)
train_speech_raw = train_data['speech']  #Input (3, num ,240,73)

# train_X = train_X[:-1:10,:,:]
# train_Y = train_Y[:-1:10,:]

test_data = np.load(datapath + test_dblist[0] + '.npz')
test_body_raw = test_data['body']  #Input (1044,240,73)
test_face_raw = test_data['face']  #Input (1044,240,73)
test_speech_raw = test_data['speech']  #Input (1044,240,73)

######################################
# Input data selection

if args.inputSubject == 2:

    train_face = np.concatenate( (train_face_raw[2,:,:,:], train_face_raw[1,:,:,:]),axis=0)
    test_face = np.concatenate( (test_face_raw[2,:,:,:], test_face_raw[1,:,:,:]),axis=0)

    train_body = np.concatenate( (train_body_raw[2,:,:,:], train_body_raw[1,:,:,:]),axis=0)
    test_body = np.concatenate( (test_body_raw[2,:,:,:], test_body_raw[1,:,:,:]),axis=0)

    train_X = np.concatenate( (train_face, train_body),axis=2)
    test_X = np.concatenate( (test_face, test_body),axis=2)

    # #Own Body - Speak
    train_Y = np.concatenate( (train_speech_raw[2], train_speech_raw[1]), axis=0)  #Input (3, num,240,73)
    test_Y = np.concatenate( (test_speech_raw[2],test_speech_raw[1]),axis=0)   #Input (3, num,240,73)

else:
    print( "args.inputSubject: {}".format(args.inputSubject) )
    #train_X = train_X[args.inputSubject,:,:,:]
    #test_X = test_X[args.inputSubject,:,:,:]

    if args.inputSubject==1:
        
        train_face = np.concatenate( (train_face_raw[2,:,:,:], train_face_raw[1,:,:,:]),axis=0)
        test_face = np.concatenate( (test_face_raw[2,:,:,:], test_face_raw[1,:,:,:]),axis=0)

        train_body = np.concatenate( (train_body_raw[2,:,:,:], train_body_raw[1,:,:,:]),axis=0)
        test_body = np.concatenate( (test_body_raw[2,:,:,:], test_body_raw[1,:,:,:]),axis=0)

        train_X = np.concatenate( (train_face, train_body),axis=2)
        test_X = np.concatenate( (test_face, test_body),axis=2)

        #Reverse
        train_Y = np.concatenate( (train_speech_raw[1], train_speech_raw[2]), axis=0)  #Input (3, num,240,73)
        test_Y = np.concatenate( (test_speech_raw[1],test_speech_raw[2]),axis=0)   #Input (3, num,240,73)

    #else: #args.inputSubject==0:
        # train_X = np.concatenate( (train_X_raw[0,:,:,:], train_X_raw[0,:,:,:]),axis=0)
        # test_X = np.concatenate( (test_X_raw[0,:,:,:], test_X_raw[0,:,:,:]),axis=0)

        # #Reverse
        # train_Y = np.concatenate( (train_Y_raw[1], train_Y_raw[2]), axis=0)  #Input (3, num,240,73)
        # test_Y = np.concatenate( (test_Y_raw[1],test_Y_raw[2]),axis=0)   #Input (3, num,240,73)

    #train_Y = train_Y[2]  #Input (3, num,240,73)
    #test_Y = test_Y[2]  #Input (3, num,240,73)


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

lstm_hidden_dim = 20
feature_dim = 73
#model = modelZoo.naive_lstm(batch_size, lstm_hidden_dim, feature_dim).cuda()
#model=modelZoo.regressor_fcn_bn().cuda()
#model = getattr(modelZoo,args.model)().cuda()
model = modelZoo.speackClass_allSignal().cuda()
#model = modelZoo.speackClass_face().cuda()
model.train()

# for param in model.parameters():
#     print(type(param.data), param.size())

# Loss Function #
criterion = nn.BCELoss()

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
train_X = np.swapaxes(train_X, 1, 2).astype(np.float32) #(num,featureDim, frames)
train_Y = train_Y.astype(np.float32)

test_X = np.swapaxes(test_X, 1, 2).astype(np.float32) #(num, featureDim, frames)
test_Y = test_Y.astype(np.float32)


######################################
# Compute mean and std 
feet = np.array([12,13,14,15,16,17,24,25,26,27,28,29])
feet = feet+5   #initial 5dim is for face

Xmean = train_X.mean(axis=2).mean(axis=0)[np.newaxis,:,np.newaxis]  #(1, 73, 1)


Xmean[:,-7:-4] = 0.0
Xmean[:,-4:]   = 0.5

#Xstd = np.array([[[train_X.std()]]]).repeat(train_X.shape[1], axis=1) #(1, 73, 1)


Xstd = np.array([[[ train_X[:,5:].std()]]]).repeat(train_X.shape[1], axis=1) #(1, 73, 1)

Xstd[:,feet]  = 0.9 * Xstd[:,feet]
Xstd[:,-7:-5] = 0.9 * train_X[:,-7:-5].std()
Xstd[:,-5:-4] = 0.9 * train_X[:,-5:-4].std()
Xstd[:,-4:]   = 0.5

#Face part
Xstd[:,:5] = train_X[:,:5].std()

# Save mean and var
np.savez_compressed(checkpointFolder+'/preprocess_core.npz', Xmean=Xmean, Xstd=Xstd)

# Data standardization 
train_X = (train_X - Xmean) / Xstd
test_X = (test_X - Xmean) / Xstd



# Data blocking

# dataBlockingMode = 0     #No block
# dataBlockingMode = 1     #No face: body only
# dataBlockingMode = 2     #No body: face only
dataBlockingMode = args.blockmode 
if dataBlockingMode==1:     #No face: body only
    logger.info('###: Blocking face features: {0}')
    train_X[:,:5,:] = 0   #train_X= (batchsize,featureNum,frames)
    test_X[:,:5,:] = 0   #train_X= (batchsize,featureNum,frames)
elif dataBlockingMode==2: #No body: face only
    logger.info('###: Blocking body features: {0}')
    train_X[:,5:,:] = 0   #train_X= (batchsize,featureNum,frames)
    test_X[:,5:,:] = 0   #train_X= (batchsize,featureNum,frames)


# Data Shuffle
I = np.arange(len(train_X))
rng.shuffle(I); 
train_X = train_X[I]
train_Y = train_Y[I]


# # Random person Shuffle Input
# I = np.arange(len(test_X))
# rng.shuffle(I); 
# test_X = test_X[I]

logger.info('Input data size: {0}'.format(train_X.shape))


######################################
# Some settings before training
if train_X.shape[0]  < batch_size:
    batch_size = train_X.shape[0]
curBestAccu = 0
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
        inputData_np = train_X[idxStart:(idxStart+batch_size),:,:]        #(batchsize,featureNum,frames)

        #Reordering from (batchsize,featureNum,frames) ->(batch, frame,features)
        outputData_np = train_Y[idxStart:(idxStart+batch_size),:]       #(batch,  frame, features)
        #outputData_np = outputData_np[:,:,np.newaxis]   #(batch, frame, 1)
        outputData_np = outputData_np[:,np.newaxis,:]   #(batch, frame, 1)

        #numpy to tensor
        if bAddNoise:
            #Add noise
            inputData = Variable(torch.from_numpy(inputData_np))
            noise = torch.randn_like(inputData)
            inputData = inputData + noise #* 2.0
            inputData = inputData.cuda()
        else:
            inputData = Variable(torch.from_numpy(inputData_np)).cuda()
        outputGT = Variable(torch.from_numpy(outputData_np)).cuda()
    

        # ===================forward=====================
        #model.hidden = model.init_hidden() #clear out hidden state of the LSTM
        output = model(inputData) #output: (batch, frames, 1)
        loss = criterion(output, outputGT)

        l1_reg = None
        for W in model.parameters():
            if l1_reg is None:
                l1_reg = W.norm(1)
            else:
                l1_reg = l1_reg + W.norm(1)        
        l1_regularization = 0.001 * l1_reg
        loss = loss + l1_regularization

        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================log========================
        avgLoss += loss.item()*batch_size

        # compute accuracy
        correct = (outputGT[:,0,:].eq( (output[:,0,:]>0.5).float() )).sum() #Just check the last one
        acc += correct.item()
        cnt += batch_size * outputGT.shape[-1]  # batchNum x frameNum

        
        if tensorboard_bLog:
            # 1. Log scalar values (scalar summary)
            if int(torch.__version__[2])==2:
                info = { 'loss': loss.data[0] }
            else:
                info = { 'loss': loss.item(), 'acc': correct.item() }

            for tag, value in info.items():
                tb_logger.scalar_summary(tag, value, stepNum)
        
        stepNum = stepNum+1

    ######################################
    # Logging
    temp_str = 'model: {}, epoch [{}/{}], avg loss:{:.4f}, accuracy:{:.4f}'.format(checkpointFolder_base, epoch +pretrain_epoch, num_epochs, avgLoss/ (len(batchinds)*batch_size), float(acc)/cnt )
    logger.info(temp_str)


    ######################################
    # Check Testing Error
    batch_size_test = batch_size
    test_loss = 0.0
    acc= 0.0
    cnt =0.0

    model.eval()
    batchinds = np.arange(test_X.shape[0] // batch_size_test)
    for bii, bi in enumerate(batchinds):

        idxStart  = bi*batch_size_test
        inputData_np = test_X[idxStart:(idxStart+batch_size),:,:]      

        #Reordering from (batchsize,featureNum,frames) ->(batch, frame,features)
        #inputData_np = np.swapaxes(inputData_np, 1, 2) #(batch, frame,features)
        outputData_np = test_Y[idxStart:(idxStart+batch_size),:]      
        #outputData_np = outputData_np[:,:,np.newaxis]
        outputData_np = outputData_np[:,np.newaxis, :]

        #numpy to Tensors
        inputData = Variable(torch.from_numpy(inputData_np)).cuda()
        outputGT = Variable(torch.from_numpy(outputData_np)).cuda()
    

        #model.hidden = model.init_hidden() #clear out hidden state of the LSTM
        output = model(inputData)

        # l1_reg = None
        # for W in model.parameters():
        #     if l1_reg is None:
        #         l1_reg = W.norm(1)
        #     else:
        #         l1_reg = l1_reg + W.norm(1)        
        # l1_regularization = 0.1 * l1_reg


        loss = criterion(output, outputGT)#+ l1_regularization

        test_loss += loss.item()*batch_size_test
        #test_loss += loss.data.cpu().numpy().item()* batch_size_test # sum up batch loss

        # compute accuracy
        correct = (outputGT[:,0,:].eq( (output[:,0,:]>0.5).float() )).sum() #Just check the last one
        acc += correct.item()
        cnt += batch_size * outputGT.shape[-1]  # batchNum x frameNum


    test_loss /= len(batchinds)*batch_size_test
    newAcc = float(acc)/cnt
    
    logger.info('    On testing data: average loss: {:.4f}, Accuracy: {:.4f} (best {:.4f})\n'.format(test_loss, acc/cnt,curBestAccu))
    if tensorboard_bLog:
        info = { 'test_acc': newAcc, 'test_loss': test_loss }
        for tag, value in info.items():
            tb_logger.scalar_summary(tag, value, stepNum)
        
    bNewBest = False
    if curBestAccu<newAcc:
        curBestAccu = newAcc
        bNewBest = True

    ######################################
    # Save parameters, if current results are the best
    if bNewBest or (epoch + pretrain_epoch) % args.checkpoint_freq == 0:
    #if (epoch + pretrain_epoch) % 1 == 0:
        fileName = checkpointFolder+ '/checkpoint_e' + str(epoch + pretrain_epoch) + '_acc{:.4f}'.format(newAcc) + '.pth'
        torch.save(model.state_dict(), fileName)
        fileName = checkpointFolder+ '/opt_state.pth'    #overwrite
        torch.save(optimizer.state_dict(), fileName)
        #torch.save(model, fileName)
