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

######################################3
# Parameter Handling
parser = my_args_parser()
args = parser.parse_args()
#args.batch = 50


# Some initializations #
torch.cuda.set_device(args.gpu)

rng = np.random.RandomState(23456)
torch.manual_seed(23456)
torch.cuda.manual_seed(23456)


######################################
# Dataset 
#datapath ='/ssd/codes/pytorch_motionSynth/motionsynth_data' 
datapath ='../../motionsynth_data/data/processed/' 

# train_dblist = ['data_hagglingSellers_speech_face_60frm_5gap_white_training']
# test_dblist = ['data_hagglingSellers_speech_face_60frm_5gap_white_testing']

train_dblist = ['data_hagglingSellers_speech_group_face_30frm_10gap_white_training']
test_dblist = ['data_hagglingSellers_speech_group_face_30frm_10gap_white_testing']
#train_dblist = ['data_hagglingSellers_speech_group_face_30frm_5gap_white_training_tiny']
#test_dblist = ['data_hagglingSellers_speech_group_face_30frm_5gap_white_training_tiny']

train_data = np.load(datapath + train_dblist[0] + '.npz')

train_X_raw = train_data['clips']  #Input (3, numClip, chunkLengh, featureDim:200) 
train_speech_raw = train_data['speech']  #Input (3, numClip, chunkLengh)

# #Attach speech signal
# train_X_raw = train_X_raw[:,:,:,:1]
# train_speech_raw = np.expand_dims(train_speech_raw,3)
# train_X_raw = np.concatenate((train_X_raw,train_speech_raw),axis=3)


#Check Bad Face parameters (abs(values)>2)
bDebug_dataCheck = False
if bDebug_dataCheck:
    badClipIndex=[]
    for pIdx in range(train_X_raw.shape[0]):
        bad_test = train_X_raw[pIdx]
        bad_test = np.max(np.max(bad_test,axis=2),axis=1)

        import matplotlib.pyplot as plt
        plt.plot(bad_test)
        plt.show()

    ##Debug: visualize
    sys.path.append('/ssd/codes/glvis_python/')
    import glViewer
    testParam = train_X_raw[2,[9482, 11381, 15670, 15671, 15672, 17707],:,:100]
    testParam = np.reshape(testParam,(-1,100))
    testParam =np.swapaxes(testParam,0,1)

    glViewer.SetFaceParmData([testParam])
    glViewer.init_gl()


# train_X = train_X[:-1:10,:,:]
# train_Y = train_Y[:-1:10,:]

test_data = np.load(datapath + test_dblist[0] + '.npz')
test_X_raw = test_data['clips']  #Input (3, numClip, chunkLengh, featureDim:200)
test_speech_raw = test_data['speech']  #Input (3, numClip, chunkLengh)


#Attach speech signal
test_X_raw = test_X_raw[:,:,:,:1]
test_speech_raw = np.expand_dims(test_speech_raw,3)
test_X_raw = np.concatenate((test_X_raw,test_speech_raw),axis=3)


######################################
# Input/Output Option
train_X = train_X_raw[1,:,:,:]      #(num, chunkLength, dim:200) //person 1 (first seller's value)
train_Y = train_X_raw[2,:,:,:]    #2nd seller's data

train_X_concat = np.concatenate((train_X,train_Y),axis=0)
train_Y_concat = np.concatenate((train_Y,train_X),axis=0)

test_X = test_X_raw[1,:,:,:]      #(num, chunkLength, dim:200) //person 1 (first seller's value)
test_Y = test_X_raw[2,:,:,:]


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
model = modelZoo.naive_lstm(batch_size).cuda()
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
train_X = np.swapaxes(train_X, 1, 2).astype(np.float32) #(num, frames, dim:200) => (num, 200, frames)
train_Y = np.swapaxes(train_Y, 1, 2).astype(np.float32) #(num, frames, dim:200) => (num, 200, frames)
#train_Y = train_Y.astype(np.float32)

test_X = np.swapaxes(test_X, 1, 2).astype(np.float32) #(num, 200, 1)
test_Y = np.swapaxes(test_Y, 1, 2).astype(np.float32) #(num, 200, 1)
#test_Y = test_Y.astype(np.float32)

# Compute mean and var
Xmean = train_X.mean(axis=2).mean(axis=0)[np.newaxis,:,np.newaxis] #(1, 200, 1)
Xstd = np.array([[[train_X.std()]]]).repeat(train_X.shape[1], axis=1) #(1, 200, 1)

Ymean = train_Y.mean(axis=2).mean(axis=0)[np.newaxis,:,np.newaxis] #(1, 200, 1)
Ystd = np.array([[[train_Y.std()]]]).repeat(train_Y.shape[1], axis=1) #(1, 200, 1)

# Save mean and var
np.savez_compressed(checkpointFolder+'/preprocess_core.npz', Xmean=Xmean, Xstd=Xstd , Ymean=Ymean, Ystd=Ystd)

# Data standardization 
train_X = (train_X - Xmean) / Xstd
test_X = (test_X - Xmean) / Xstd

train_Y = (train_Y - Ymean) / Ystd
test_Y = (test_Y - Ymean) / Ystd

# Data Shuffle
I = np.arange(len(train_X))
rng.shuffle(I); 
train_X = train_X[I]
train_Y = train_Y[I]

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
faceFeatureDim = 2
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
        inputData_np = train_X[idxStart:(idxStart+batch_size),:faceFeatureDim,:]

        #Reordering from (batchsize,featureNum,frames) ->(batch, frame,features)
        inputData_np = np.swapaxes(inputData_np, 1, 2) #(batch,  frame, features)
        outputData_np = train_Y[idxStart:(idxStart+batch_size),:faceFeatureDim,:]      
        outputData_np = np.swapaxes(outputData_np, 1, 2) #(batch,  frame, features)
        #outputData_np = outputData_np[:,:,np.newaxis]   #(batch, frame, 1)

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
        model.hidden = model.init_hidden() #clear out hidden state of the LSTM
        output = model(inputData) #output: (batch, frames, 1)
        loss = criterion(output, outputGT)

        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================log========================
        avgLoss += loss.item()*batch_size

        """
        if(loss.item()>10) and epoch>10:
        #if epoch>10 and idxStart >100:
        #if idxStart >100:
            print("idxStart: {}, loss:{}".format(idxStart,loss.item()))

            # ##Debug
            inputData_np_debug = np.swapaxes(inputData_np,1,2)
            inputData_np_debug = inputData_np_debug*Xstd[:,:100,:] + Xmean[:,:100,:]
            inputData_np_debug =np.swapaxes(inputData_np_debug,1,2)
            testParam = np.reshape(inputData_np_debug,(-1,100))
            testParam =np.swapaxes(testParam,0,1)
            

            glViewer.SetFaceParmData([testParam])
            glViewer.init_gl()
        """

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

    #continue
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
        inputData_np = test_X[idxStart:(idxStart+batch_size),:faceFeatureDim,:]      

        #Reordering from (batchsize,featureNum,frames) ->(batch, frame,features)
        inputData_np = np.swapaxes(inputData_np, 1, 2) #(batch, frame,features)
        outputData_np = test_Y[idxStart:(idxStart+batch_size),:faceFeatureDim,:]      
        #outputData_np = outputData_np[:,:,np.newaxis]
        outputData_np = np.swapaxes(outputData_np, 1, 2) #(batch,  frame, features)

        #numpy to Tensors
        inputData = Variable(torch.from_numpy(inputData_np)).cuda()
        outputGT = Variable(torch.from_numpy(outputData_np)).cuda()
    

        model.hidden = model.init_hidden() #clear out hidden state of the LSTM
        output = model(inputData) #output: (batch, frames, 1)
        loss = criterion(output, outputGT)

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
    #if bNewBest:#(epoch + pretrain_epoch) % args.checkpoint_freq == 0:
    if (epoch + pretrain_epoch) % 100 == 0:
        fileName = checkpointFolder+ '/checkpoint_e' + str(epoch + pretrain_epoch) + '_loss{:.4f}'.format(test_loss) + '.pth'
        torch.save(model.state_dict(), fileName)
        fileName = checkpointFolder+ '/opt_state.pth'    #overwrite
        torch.save(optimizer.state_dict(), fileName)
        #torch.save(model, fileName)
