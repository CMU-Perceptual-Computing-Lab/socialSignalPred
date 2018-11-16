import os
import sys
import numpy as np
import scipy.io as io
import cPickle as pickle

# from Quaternions import Quaternions
# from Pivots import Pivots

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
import utility
from utility import print_options,save_options
from utility import setCheckPointFolder
from utility import my_args_parser


######################################
# Parameter Handling
parser = my_args_parser()
args = parser.parse_args()

######################################
# Manual Parameter Setting
#args.model ='regressor_holden'
# args.model ='regressor_fcn'
# args.model ='regressor_fcn_bn'
# args.model ='regressor_fcn_bn_encoder'
# args.model ='regressor_fcn_bn_encoder_2'



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

# train_dblist = ['data_hagglingSellers_speech_body_120frm_5gap_white_training']
# test_dblist = ['data_hagglingSellers_speech_body_120frm_10gap_white_testing']

train_dblist = ['data_hagglingSellers_speech_body_group_240frm_30gap_white_noGa_training']
test_dblist = ['data_hagglingSellers_speech_body_group_240frm_15gap_white_noGa_testing']


train_dblist = ['data_hagglingSellers_speech_body_group_120frm_10gap_white_noGa_training']
test_dblist = ['data_hagglingSellers_speech_body_group_120frm_30gap_white_noGa_testing']

# pkl_file = open(datapath + train_dblist[0] + '.pkl', 'rb')
# train_data = pickle.load(pkl_file)
# pkl_file.close()
train_data = np.load(datapath + train_dblist[0] + '.npz')
train_X_raw= train_data['data']  #Input (numClip, frames, featureDim:73)
#train_speech_raw = train_data['speech']  #Input (numClip, frames)

test_data = np.load(datapath + test_dblist[0] + '.npz')
# pkl_file = open(datapath + test_dblist[0] + '.pkl', 'rb')
# test_data = pickle.load(pkl_file)
# pkl_file.close()
test_X_raw= test_data['data']      #Input (numClip, frames, featureDim:73)
#test_speech_raw = test_data['speech']    #Input (numClip, frames)

logger.info("Raw: Training Dataset: {}".format(train_X_raw.shape))
#Select speaking time only only
# speak_time =[]
# #Choose only speaking signal
# for i in range(train_X_raw.shape[0]):
#     speechSignal = train_speech_raw[i,:]
#     if np.min(speechSignal)==1:
#         speak_time.append(i)
# train_X_raw = train_X_raw[speak_time,:,:]
# logger.info("Training Dataset: {}".format(train_X_raw.shape))


"""Visualize X and Y
#by jhugestar
for frameIdx in range(1,train_X_raw.shape[1],10):
    sys.path.append('/ssd/codes/glvis_python/')
    #from glViewer import showSkeleton,show_Holden_Data_73 #opengl visualization 
    import glViewer
    glViewer.show_Holden_Data_73([ np.swapaxes(train_X_raw[1,frameIdx,:,:],0,1), np.swapaxes(train_X_raw[2,frameIdx,:,:],0,1) ] )
"""

############################################################################
# Load Pretrained Autoencoder

######################################
# Checkout Folder and pretrain file setting
checkpointRoot = './'
ae_checkpointFolder = checkpointRoot+ '/social_autoencoder_first_try9_120frm_best_noReg/'
preTrainFileName= 'checkpoint_e1009_loss0.0085.pth'


# ######################################
# # Load Pretrained Auto-encoder
ae_preprocess = np.load(ae_checkpointFolder + 'preprocess_core.npz') #preprocess['Xmean' or 'Xstd']: (1, 73,1))
ae_model = modelZoo.autoencoder_first().cuda()

#Creat Model
trainResultName = ae_checkpointFolder + preTrainFileName
loaded_state = torch.load(trainResultName, map_location=lambda storage, loc: storage)

ae_model.load_state_dict(loaded_state,strict=False) #strict False is needed some version issue for batchnorm
ae_model = ae_model.eval()  #Do I need this again?

# Freeze au layers
for p in ae_model.parameters():
    p.requires_grad_(False)


######################################
# Network Setting
num_epochs = args.epochs #500
#batch_size = 128
batch_size = args.batch
learning_rate = 1e-3

model = getattr(modelZoo,args.model)().cuda()
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

train_X = train_X_raw[0,:,:,:]      #(num, frameNum, featureDim:3)
train_X = np.concatenate( (train_X, train_X_raw[1,:,:,:]), axis= 2)    #(num, chunkLength, 18)
train_Y = train_X_raw[2,:,:,:]    #2nd seller's position only

## Change the second and third 
train_X_swap = train_X_raw[0,:,:,:]      #(num, chunkLength, 9) //person0,1's all values (position, head orientation, body orientation)
train_X_swap = np.concatenate( (train_X_swap, train_X_raw[2,:,:,:]), axis= 2)    #(num, chunkLength, 18)
train_Y_swap = train_X_raw[1,:,:,:]    #2nd seller's position only

train_X = np.concatenate( (train_X, train_X_swap), axis=0)
train_Y = np.concatenate( (train_Y, train_Y_swap), axis=0)


## Test data
test_X = test_X_raw[0,:,:,:]      #(num, chunkLength, 9) //person0,1's all values (position, head orientation, body orientation)
test_X = np.concatenate( (test_X, test_X_raw[1,:,:,:]), axis= 2)      #(num, chunkLength, 18)
test_Y = test_X_raw[2,:,:,:]    #2nd seller's position only

test_X_swap = test_X_raw[0,:,:,:]      #(num, chunkLength, 9) //person0,1's all values (position, head orientation, body orientation)
test_X_swap = np.concatenate( (test_X_swap, test_X_raw[2,:,:,:]), axis= 2)      #(num, chunkLength, 18)
test_Y_swap = test_X_raw[1,:,:,:]    #2nd seller's position only

test_X = np.concatenate( (test_X, test_X_swap), axis=0)
test_Y = np.concatenate( (test_Y, test_Y_swap), axis=0)



######################################
# Data pre-processing
train_X = np.swapaxes(train_X, 1, 2).astype(np.float32) #(num, chunkLength, featureDim) ->(num, featureDim, chunkLength)
train_Y = np.swapaxes(train_Y, 1, 2).astype(np.float32) #(num, chunkLength, featureDim) ->(num, featureDim, chunkLength)

test_X = np.swapaxes(test_X, 1, 2).astype(np.float32) 
test_Y = np.swapaxes(test_Y, 1, 2).astype(np.float32) 


# Standardization (consider seller motion only)
feet = np.array([12,13,14,15,16,17,24,25,26,27,28,29])

train_body = train_X[:,:73,:]
body_mean = train_body.mean(axis=2).mean(axis=0)[np.newaxis,:,np.newaxis]  #(1, 73, 1)
body_mean[:,-7:-4] = 0.0
body_mean[:,-4:]   = 0.5

body_std = np.array([[[train_body.std()]]]).repeat(train_body.shape[1], axis=1) #(1, 73, 1)
body_std[:,feet]  = 0.9 * body_std[:,feet]
body_std[:,-7:-5] = 0.9 * train_body[:,-7:-5].std()
body_std[:,-5:-4] = 0.9 * train_body[:,-5:-4].std()
body_std[:,-4:]   = 0.5


body_mean_two = np.concatenate((body_mean,body_mean),axis=1)
body_std_two = np.concatenate((body_std,body_std),axis=1)
# Data standardization 
train_X = (train_X - body_mean_two) / body_std_two
test_X = (test_X - body_mean_two) / body_std_two

# Data standardization : for trajectory
train_Y = (train_Y - body_mean) / body_std
test_Y = (test_Y - body_mean) / body_std

# Save mean and var
np.savez_compressed(checkpointFolder+'/preprocess_core.npz', body_mean=body_mean, body_std=body_std, body_mean_two=body_mean_two, body_std_two=body_std_two)


# Data Shuffle
I = np.arange(len(train_X))
rng.shuffle(I)
train_X = train_X[I]
train_Y = train_Y[I]
#train_Y = train_Y[I]

logger.info('Input data size: {0}'.format(train_X.shape))



######################################
# Debug: Visualization
"""Visualize X and Y"""
# sys.path.append('/ssd/codes/glvis_python/')
# import glViewer
# for sampleIdx in range(train_body.shape[0]):
    
#     vis_body = train_body[sampleIdx:(sampleIdx+1),:,:] # (featureDim, frames)
#     vis_body = (vis_body* body_std) + body_mean
#     #vis_body = vis_body[0,:,:]
#     vis_body = vis_body[0,:-4,:] #ignore foot step info
    
#     vis_traj = train_traj[sampleIdx:(sampleIdx+1),:,:] # (featureDim, frames)
#     vis_traj = (vis_traj* traj_std) + traj_mean
#     vis_traj = vis_traj[0,:,:]

#     glViewer.set_Holden_Data_73([ vis_body] )
#     glViewer.set_Holden_Trajectory_3([ vis_traj] )

#     glViewer.init_gl()
    

######################################
# Some settings before training
if train_X.shape[0]  < batch_size:
    batch_size = train_X.shape[0]
curBestloss = 1e5
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
    cnt = 0
    for bii, bi in enumerate(batchinds):

        idxStart  = bi*batch_size
        inputData_np = train_X[idxStart:(idxStart+batch_size),:,:]  #(batch, 3, frameNum)
        #outputData_np = train_body[idxStart:(idxStart+batch_size),:-4,:] #(batch, 73 - 4, frameNum)
        outputData_np = train_Y[idxStart:(idxStart+batch_size),:,:] #(batch, 73 - 4, frameNum)

        inputData = Variable(torch.from_numpy(inputData_np)).cuda()  #(batch, 3, frameNum)
        outputGT = Variable(torch.from_numpy(outputData_np)).cuda()  #(batch, 73, frameNum)

        # ===================forward=====================
        #output= model(inputData)
        output= model(inputData)
        output = ae_model.decoder(output)

        #Amplify the loss for the angle layer
        # output[:,-5,:] *= 10
        # outputGT[:,-5,:] *= 10
        loss = criterion(output, outputGT)
        #loss = criterion(output, inputData)
        #loss, recon_loss, kld_loss = modelZoo.vae_loss_function(output, inputData, mu, logvar,criterion,args.weight_kld)

        # l1_reg = None
        # for W in model.parameters():
        #     if l1_reg is None:
        #         l1_reg = W.norm(1)
        #     else:
        #         l1_reg = l1_reg + W.norm(1)        
        # l1_regularization = 0.1 * l1_reg

        loss = loss #+ l1_regularization

        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================log========================
        # print('model: {}, epoch [{}/{}], loss:{:.4f}'
        #             .format(checkpointFolder_base, epoch +pretrain_epoch, num_epochs, loss.item()))
        avgLoss += loss.item()*batch_size
    
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
    # test_avgReconLoss = 0
    # test_avgKLDLoss = 0
    cnt =0.0

    model.eval()
    batchinds = np.arange(test_X.shape[0] // batch_size_test)
    for bii, bi in enumerate(batchinds):

        idxStart  = bi*batch_size 
        inputData_np = test_X[idxStart:(idxStart+batch_size),:,:]  #(batch, 3, frameNum)
        #outputData_np = test_body[idxStart:(idxStart+batch_size),:-4,:] #(batch, 73 - 4, frameNum)
        outputData_np = test_Y[idxStart:(idxStart+batch_size),:,:] #(batch, 73 - 4, frameNum)

        inputData = Variable(torch.from_numpy(inputData_np)).cuda()  #(batch, 3, frameNum)
        outputGT = Variable(torch.from_numpy(outputData_np)).cuda()  #(batch, 73, frameNum)


        # ===================forward=====================
        #output = model(inputData)
        output= model(inputData)
        output = ae_model.decoder(output)

        loss = criterion(output, outputGT)
        test_loss += loss.item()*batch_size_test


    test_loss /= len(batchinds)*batch_size_test
    
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
    if bNewBest or (epoch + pretrain_epoch) % args.checkpoint_freq == 0  or ( (epoch + pretrain_epoch<50) and ((epoch + pretrain_epoch)%10 ==0) ):
    #if (epoch + pretrain_epoch) % 1 == 0:
        fileName = checkpointFolder+ '/checkpoint_e' + str(epoch + pretrain_epoch) + '_loss{:.4f}'.format(test_loss) + '.pth'
        torch.save(model.state_dict(), fileName)
        fileName = checkpointFolder+ '/opt_state.pth'    #overwrite
        torch.save(optimizer.state_dict(), fileName)
        #torch.save(model, fileName)
