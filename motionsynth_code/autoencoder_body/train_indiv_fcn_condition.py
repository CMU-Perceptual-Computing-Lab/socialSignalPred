import os
import sys
import numpy as np
import scipy.io as io


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
# Manual Parameter Setting
#args.model ='autoencoder_first_speakConditional'
#args.solver = 'sgd'
#args.finetune = 'social_autoencoder_3conv_vae'
#args.check_root = '/posefs2b/Users/hanbyulj/pytorch_motionSynth/checkpoint'
#args.batch = 2
#args.weight_kld = 0.0001

# Some initializations #
torch.cuda.set_device(args.gpu)

rng = np.random.RandomState(23456)
torch.manual_seed(23456)
torch.cuda.manual_seed(23456)


######################################
# Dataset 
#datapath ='/ssd/codes/pytorch_motionSynth/motionsynth_data' 
datapath ='../../motionsynth_data/data/processed/' 

#train_dblist = ['data_hagglingSellers_speech_formation_30frm_5gap_white_training']
#train_dblist = ['data_hagglingSellers_speech_body_120frm_10gap_white_training']
train_dblist = ['data_hagglingSellers_speech_body_120frm_10gap_white_noGa_training']
test_dblist = ['data_hagglingSellers_speech_body_120frm_10gap_white_noGa_testing']

train_data = np.load(datapath + train_dblist[0] + '.npz')
train_X_raw= train_data['clips']  #Input (numClip, frames, featureDim:73)
train_speech_raw = train_data['speech']  #Input (numClip, frames)

test_data = np.load(datapath + test_dblist[0] + '.npz')
test_X_raw= test_data['clips']      #Input (numClip, frames, featureDim:73)
test_speech_raw = test_data['speech']    #Input (numClip, frames)

logger.info("Raw: Training Dataset: {}".format(train_X_raw.shape))
#Select speaking time only only
speak_time =[]
# #Choose only speaking signal
# logger.info("Choose speaking motion only")
# for i in range(train_X_raw.shape[0]):
#     speechSignal = train_speech_raw[i,:]
#     if np.min(speechSignal)==1:
#         speak_time.append(i)
# train_X_raw = train_X_raw[speak_time,:,:]
# logger.info("Speaking Only: Training Dataset: {}".format(train_X_raw.shape))

""" Generate Binary Speak label for each chunk """
for i in range(train_speech_raw.shape[0]):       #train_X_raw: (numClip, frames, featureDim:73)
    speechSignal = train_speech_raw[i,:]    #Input (numClip, frames)
    if np.max(speechSignal)==1:
        train_speech_raw[i,:] = 1       #If there exists, at least one speak signal, put 1
    else:
        train_speech_raw[i,:] = 0       #Otherwise. just zeros
        
for i in range(test_speech_raw.shape[0]):       #train_X_raw: (numClip, frames, featureDim:73)
    speechSignal = test_X_raw[i,:]    #Input (numClip, frames)
    if np.max(speechSignal)==1:
        test_speech_raw[i,:] = 1       #If there exists, at least one speak signal, put 1
    else:
        test_speech_raw[i,:] = 0       #Otherwise. just zeros

"""Visualize X and Y
#by jhugestar
for frameIdx in range(1,train_X_raw.shape[1],10):
    sys.path.append('/ssd/codes/glvis_python/')
    #from glViewer import showSkeleton,show_Holden_Data_73 #opengl visualization 
    import glViewer
    glViewer.show_Holden_Data_73([ np.swapaxes(train_X_raw[1,frameIdx,:,:],0,1), np.swapaxes(train_X_raw[2,frameIdx,:,:],0,1) ] )
"""

######################################
# Network Setting
num_epochs = args.epochs #500
#batch_size = 128
batch_size = args.batch
learning_rate = 1e-3

# if args.autoreg ==1: #and "vae" in args.model:
#     model = getattr(modelZoo,args.model)(frameLeng=160).cuda()
# else:
model = getattr(modelZoo,args.model)().cuda()
#model = modelZoo.autoencoder_first().cuda()
#featureDim = train_X_raw.shape[2]
#latentDim = 200
#model = modelZoo.autoencoder_3conv_vect_vae(featureDim,latentDim).cuda()
model.train()

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
# Input/Output Option
train_X = train_X_raw#[1,:,:,:]      #1st seller, (num, frameNum, featureDim:73)
train_X_speech = np.expand_dims(train_speech_raw,2) #(num, frameNum, 1)

test_X = test_X_raw#[1,:,:,:]      #1st seller, (num, frameNum, featureDim:73)
test_X_speech = np.expand_dims(test_speech_raw, 2) #(num, frameNum, 1)


# Compute mean and std 
train_X = np.swapaxes(train_X, 1, 2).astype(np.float32) #(num, 73, frameNum)
train_X_speech = np.swapaxes(train_X_speech, 1, 2).astype(np.float32)       #(num, 1, frameNum)
#train_Y = np.swapaxes(train_Y, 1, 2).astype(np.float32) #(num, 73, frameNum)

test_X = np.swapaxes(test_X, 1, 2).astype(np.float32) #(num, 73, frameNum)
test_X_speech = np.swapaxes(test_X_speech, 1, 2).astype(np.float32)     #(num, 1, frameNum)
#test_Y = np.swapaxes(test_Y, 1, 2).astype(np.float32) #(num, 73, frameNum)

data_all = np.concatenate( (train_X, train_X), axis= 0)
feet = np.array([12,13,14,15,16,17,24,25,26,27,28,29])

Xmean = data_all.mean(axis=2).mean(axis=0)[np.newaxis,:,np.newaxis]  #(1, 73, 1)
Xmean[:,-7:-4] = 0.0
Xmean[:,-4:]   = 0.5

Xstd = np.array([[[data_all.std()]]]).repeat(data_all.shape[1], axis=1) #(1, 73, 1)
Xstd[:,feet]  = 0.9 * Xstd[:,feet]
Xstd[:,-7:-5] = 0.9 * data_all[:,-7:-5].std()
Xstd[:,-5:-4] = 0.9 * data_all[:,-5:-4].std()
Xstd[:,-4:]   = 0.5

# Data standardization 
train_X = (train_X - Xmean) / Xstd
#train_Y = (train_Y - Xmean) / Xstd

test_X = (test_X - Xmean) / Xstd
#test_Y = (test_Y - Xmean) / Xstd

# Save mean and var
np.savez_compressed(checkpointFolder+'/preprocess_core.npz', Xmean=Xmean, Xstd=Xstd)

# Data Shuffle
I = np.arange(len(train_X))
rng.shuffle(I)
train_X = train_X[I]
train_X_speech = train_X_speech[I]
#train_Y = train_Y[I]

logger.info('Input data size: {0}'.format(train_X.shape))

######################################
# Some settings before training
if train_X.shape[0]  < batch_size:
    batch_size = train_X.shape[0]
curBestloss = 1e3
#Compute stepNum start point (to be continuos in tensorboard if pretraine data is loaded)
filelog_str = ''
stepNum = pretrain_epoch* len(np.arange(train_X.shape[0] // pretrain_batch_size))


######################################
# Training
for epoch in range(num_epochs):

    model.train()

    batchinds = np.arange(train_X.shape[0] // batch_size)
    rng.shuffle(batchinds)
    
    # Each Batch
    avgLoss =0
    avgReconLoss = 0
    avgKLDLoss =0
    cnt = 0
    for bii, bi in enumerate(batchinds):

        idxStart  = bi*batch_size
        inputData_np = train_X[idxStart:(idxStart+batch_size),:,:]  #(batch, 73, frameNum)
        speech_np = train_X_speech[idxStart:(idxStart+batch_size),:,:]*10 #(batch, 1, frames)
        inputData_np = np.concatenate((inputData_np,speech_np),axis=1) #(batch, 73+1, frames)
        #outputData_np = train_Y[idxStart:(idxStart+batch_size),:,:]

        inputData = Variable(torch.from_numpy(inputData_np)).cuda()  #(batch, 73, frameNum)
        
        #outputGT = Variable(torch.from_numpy(outputData_np)).cuda()  #(batch, 73, frameNum)
        #outputGT = Variable(torch.from_numpy(inputData_np)).cuda()  #(batch, 73, frameNum)  

        # ===================forward=====================
        #output, mu, logvar = model(inputData)
        output = model(inputData)
        #loss = criterion(output, outputGT)

        # if args.l1regw >0.0:
        #     l1_reg = None
        #     for W in model.parameters():
        #         if l1_reg is None:
        #             l1_reg = W.norm(1)
        #         else:
        #             l1_reg = l1_reg + W.norm(1)        
        #     #l1_regularization = 0.1 * l1_reg
        #     l1_regularization = args.l1regw * l1_reg
        
        #     loss = criterion(output, inputData[:,:-1,:]) +l1_regularization
        # else:
        loss = criterion(output, inputData[:,:-1,:])

        #loss, recon_loss, kld_loss = modelZoo.vae_loss_function(output, inputData, mu, logvar,criterion,args.weight_kld)

        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================log========================
        # print('model: {}, epoch [{}/{}], loss:{:.4f}'
        #             .format(checkpointFolder_base, epoch +pretrain_epoch, num_epochs, loss.item()))
        avgLoss += loss.item()*batch_size
        # avgReconLoss += recon_loss.item()*batch_size
        # avgKLDLoss += kld_loss.item()*batch_size
    
        if tensorboard_bLog:
            info = { 'loss': loss.item()}

            for tag, value in info.items():
                tb_logger.scalar_summary(tag, value, stepNum)
    
        stepNum = stepNum+1

    ######################################
    # Logging
    temp_str = 'model: {}, epoch [{}/{}], avg loss:{:.4f}'.format(checkpointFolder_base, epoch +pretrain_epoch, num_epochs,
                                                                 avgLoss/ (len(batchinds)*batch_size)
                                                                  )
    logger.info(temp_str)
    



    ######################################
    # Check Testing Error
    batch_size_test = batch_size
    test_loss = 0
    test_avgReconLoss = 0
    test_avgKLDLoss = 0
    cnt =0.0

    model.eval()
    batchinds = np.arange(test_X.shape[0] // batch_size_test)
    for bii, bi in enumerate(batchinds):

        idxStart  = bi*batch_size
        inputData_np = test_X[idxStart:(idxStart+batch_size),:,:]
        speech_np = test_X_speech[idxStart:(idxStart+batch_size),:,:]*10 #(batch, 1, frames)
        inputData_np = np.concatenate((inputData_np,speech_np),axis=1) #(batch, 73+1, frames)
        #outputData_np = test_Y[idxStart:(idxStart+batch_size),:,:]

        inputData = Variable(torch.from_numpy(inputData_np)).cuda()  #(batch, 73, frameNum)
        #outputGT = Variable(torch.from_numpy(outputData_np)).cuda()  #(batch, 73, frameNum)
        #outputGT = Variable(torch.from_numpy(inputData_np)).cuda()  #(batch, 73, frameNum)

        # ===================forward=====================
        #output, mu, logvar = model(inputData)
        output = model(inputData)
        #loss = criterion(output, outputGT)
        loss = criterion(output, inputData[:,:-1,:])
        #loss, recon_loss, kld_loss = modelZoo.vae_loss_function(output, inputData, mu, logvar,criterion,args.weight_kld)

        test_loss += loss.item()*batch_size_test
        # test_avgReconLoss += recon_loss.item()*batch_size_test
        # test_avgKLDLoss += kld_loss.item()*batch_size_test
        #test_loss += loss.data.cpu().numpy().item()* batch_size_test # sum up batch loss


    test_loss /= len(batchinds)*batch_size_test
    test_avgReconLoss /= len(batchinds)*batch_size_test
    test_avgKLDLoss /= len(batchinds)*batch_size_test
    
    logger.info('    On testing data: average loss: {:.4f} (best {:.4f})\n'.format(test_loss, curBestloss))
    if tensorboard_bLog:
        #info = { 'test_loss': test_loss }
        info = { 'test_loss': test_loss}
        for tag, value in info.items():
            tb_logger.scalar_summary(tag, value, stepNum)
        
    bNewBest = False
    if curBestloss > test_loss:
        curBestloss = test_loss
        bNewBest = True

    ######################################
    # Save parameters, if current results are the best
    if bNewBest or (epoch + pretrain_epoch) % args.checkpoint_freq == 0:
    #if (epoch + pretrain_epoch) % 1 == 0:
        fileName = checkpointFolder+ '/checkpoint_e' + str(epoch + pretrain_epoch) + '_loss{:.4f}'.format(test_loss) + '.pth'
        torch.save(model.state_dict(), fileName)
        fileName = checkpointFolder+ '/opt_state.pth'    #overwrite
        torch.save(optimizer.state_dict(), fileName)
        #torch.save(model, fileName)
