import os
import sys
import numpy as np
import scipy.io as io
import random

"""For logging by tensorboard"""
bLog = False  
try:
	sys.path.append('../utils')
	from logger import Logger
	bLog = True
except ImportError:
	pass

import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST
import os
import argparse

import modelZoo

def print_options(parser,opt):
    option_dict = dict()

    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        option_dict[k] = v

        comment = ''
        default = parser.get_default(k)
        if v != default:
            comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)

    return message, option_dict

import json
def save_options(checkpoints_dir, message, options_dict):

    # save to the disk
    #expr_dir = os.path.join(checkpoints_dir, 'options.txt')
    #util.mkdirs(expr_dir)
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)
    file_name = os.path.join(checkpoints_dir, 'opt.txt')
    with open(file_name, 'wt') as opt_file:
        opt_file.write(message)
        opt_file.write('\n')

    file_name = os.path.join(checkpoints_dir, 'opt.json')
    with open(file_name, 'wt') as opt_file:
        json.dump(options_dict,opt_file)
        
parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--epochs', type=int, default=500001, metavar='N',
                    help='number of epochs to train (default: 50001)')

parser.add_argument('--batch', type=int, default=512, metavar='N',
                    help='batch size (default: 32)')

parser.add_argument('--gpu', type=int, default=0, metavar='N',
                    help='Select gpu (default: 0)')

parser.add_argument('--checkpoint_freq', type=int, default=500, metavar='N',
                    help='How frequently save the checkpoint (default: every 50 epoch)')

parser.add_argument('--model', type=str, default='autoencoder_first',
                    help='a model name in the model_zoo.py (default: autoencoder_first')

parser.add_argument('--solver', type=str, default='adam_ams',
                    help='Optimization solver. adam or sgd, adam_ams. (default: adam_ams')

parser.add_argument('--db', type=str, default='haggling_socialmodel_wl',
                    help='Database for training cmu...(default: cmu')

parser.add_argument('--finetune', type=str, default='',
                    help='if a folder is specified, then restart training by loading the last saved model files')

parser.add_argument('--check_root', type=str, default='./',
                    help='The root dir to make subfolders for the check point (default: ./) ')

parser.add_argument('--weight_kld', type=float, default='0.1',
                    help='Weight for the KLD term in VAE training (default: 0.1) ')

parser.add_argument('--autoreg', type=int, default='0',
                    help='If >0, train with autoregressive mode. (using init 150 frames input and later 150 frames as output) (default: 0')

args = parser.parse_args()  

#Debug
#args.model = 'autoencoder_2convLayers'
#args.model ='autoencoder_3conv_vae'
#args.model ='autoencoder_3convLayers_vect3_64'
#args.model ='autoencoder_3convLayers_vect3_2'
#args.model ='autoencoder_3convLayers_vect3_2'
#args.model ='autoencoder_3convLayers_vect'
#args.finetune = 'autoencoder_3conv_vae'
#args.check_root = '/posefs2b/Users/hanbyulj/pytorch_motionSynth/checkpoint'
#args.weight_kld = 0.01
#args.autoreg = 1     #turn on autoregressive mode
#args.db = 'edin_loco'
#args.db = 'haggling_winner_loser'

# args.model ='autoencoder_3conv_vae'
# #args.solver = 'sgd'
# args.finetune = 'social_autoencoder_3conv_vae'
# args.check_root = '/posefs2b/Users/hanbyulj/pytorch_motionSynth/checkpoint'
# args.batch = 2
#args.weight_kld = 0.0001

torch.cuda.set_device(args.gpu)

rng = np.random.RandomState(23456)
torch.manual_seed(23456)
torch.cuda.manual_seed(23456)

"""All available dataset"""
#datapath ='/ssd/codes/pytorch_motionSynth/motionsynth_data' 
datapath ='../../motionsynth_data/data/processed/' 

# train_dblist = ['data_panoptic_speech_haggling_sellers_training_byFrame']
# test_dblist = ['data_panoptic_speech_haggling_sellers_testing_byFrame']
train_dblist = ['data_panoptic_speech_haggling_sellers_training_byFrame_white_30frm_tiny']
#train_dblist = ['data_panoptic_speech_haggling_sellers_training_byFrame_white']
test_dblist = ['data_panoptic_speech_haggling_sellers_testing_byFrame_white_30frm_tiny']

train_data = np.load(datapath + train_dblist[0] + '.npz')

train_X= train_data['clips']  #Input (3700,240,73)
train_Y = train_data['classes']  #Input (3700,240,73)

train_X = train_X[:-1:10,:,:]
train_Y = train_Y[:-1:10,:]

test_data = np.load(datapath + test_dblist[0] + '.npz')
test_X= test_data['clips']  #Input (1044,240,73)
test_Y = test_data['classes']  #Input (1044,240,73)

#data_all = np.concatenate( (X,Y), axis=0) #(5366,240,73)
# X = np.swapaxes(X, 1, 2).astype(np.float32) #(num, 73, 240)
# Y = np.swapaxes(Y, 1, 2).astype(np.float32) #(num, 73, 240)

# """Visualize X and Y"""
# #by jhugestar
# for frameIdx in range(1,X.shape[0],10):
#     sys.path.append('/ssd/codes/glvis_python/')
#     from Visualize_human_gl import showSkeleton,show_Holden_Data_73 #opengl visualization 
#     show_Holden_Data_73([ X[frameIdx,:,:], Y[frameIdx,:,:]])

""" Training Network """
num_epochs = args.epochs #500
#batch_size = 128
batch_size = args.batch
learning_rate = 1e-3

# if args.autoreg ==1: #and "vae" in args.model:
#     model = getattr(modelZoo,args.model)(frameLeng=160).cuda()
# else:
#     model = getattr(modelZoo,args.model)().cuda()

import torch
from torch import nn

class naive_lstm2(nn.Module):
    def __init__(self, batch_size):
        super(naive_lstm2, self).__init__()

        self.hidden_dim = 12
        self.feature_dim= 73
        self.num_layers = 1
        self.batch_size = batch_size
        
        self.hidden = self.init_hidden()
        self.lstm = nn.LSTM(self.feature_dim, self.hidden_dim, batch_first=True) #batch_first=True makes the order as (batch, frames, featureNum)
        self.proj = nn.Linear(self.hidden_dim,1)
        self.out_act = nn.Sigmoid()

    def init_hidden(self):
        return (Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_dim)).cuda(),
                Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_dim)).cuda())
                

    #Original LSTM ordering
    # def forward(self, input_):
    #     #input_ dimension: (timestpes, batch, dim)
    #     lstm_out, self.hidden = self.lstm(
    #                 input_, self.hidden)
    #     #lstm_out:  (timesteps, batch, hidden_dim)
    #     #self.hidden (tuple with two elements):  ( (1, batch, hidden_dim),  (1, batch, hidden_dim))
        
    #     proj = self.proj(lstm_out) #input: , output:(timesteps, batch, 1)
    #     return self.out_act(proj)

    #batch_first ordering
    def forward(self, input_):

        #input_ dimension: (batch, timestpes, dim). Note I used batch_first for this ordering
        lstm_out, self.hidden = self.lstm(
                    input_, self.hidden)
        #lstm_out:  (batch, timesteps, hidden_dim)
        
        #self.hidden (tuple with two elements):  ( (1, batch, hidden_dim),  (1, batch, hidden_dim))
        proj = self.proj(lstm_out) #input:(batch, inputDim, outputDim ) -> output (batch, timesteps, )
        return self.out_act(proj)

model = naive_lstm2(batch_size).cuda()


for param in model.parameters():
    print(type(param.data), param.size())

criterion = nn.BCELoss()

if args.solver == 'adam':
    print('solver: Adam')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
elif args.solver == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    print('solver: SGD')
elif args.solver == 'adam_ams': #only for pytorch 0.4 or later.
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5, amsgrad=True)
    print('solver: Adam with AMSGrad')
else:
    print('Unknown solver option')
    assert(False)

#checkpointFolder = './autoenc_vect/'
pretrain_epoch = 0
if args.finetune =='':
    pretrain_batch_size =args.batch  #Assume current batch was used in pretraining

    if 'socialmodel' in args.db:
        checkpointFolder = args.check_root + '/social_'+ model.__class__.__name__
    else:
        checkpointFolder = args.check_root + '/'+ model.__class__.__name__
    
    if not os.path.exists(checkpointFolder):
        os.mkdir(checkpointFolder)
    else: #if already exist
        tryIdx =1
        while True:
            newCheckName = checkpointFolder + '_try' + str(tryIdx)
            if not os.path.exists(newCheckName):
                checkpointFolder = newCheckName
                os.mkdir(checkpointFolder)
                break
            else:
                tryIdx += 1
else:       #FineTuning
    checkpointFolder = args.check_root+ '/' + args.finetune

    checkpointList =  [os.path.join(checkpointFolder,f) for f in sorted(list(os.listdir(checkpointFolder)))
                if os.path.isfile(os.path.join(checkpointFolder,f))
                    and f.endswith('.pth') ] 

    #Fine Last Epoch
    last_epoch =0
    for name in checkpointList:
        pretrain_epoch= int(name[(name.find('checkpoint_') + 12):-4])
        if last_epoch <pretrain_epoch:
            last_epoch= pretrain_epoch
            trainResultName = name


    #checkpointList.sort(key=lambda x: os.path.getmtime(x))
    #trainResultName =checkpointList[-1]
    pretrain_epoch= int(trainResultName[(trainResultName.find('checkpoint_') + 12):-4])
    pretrain_epoch = pretrain_epoch+1
    
    try:
        log_file_name = os.path.join(checkpointFolder, 'opt.json')
        with open(log_file_name, 'r') as opt_file:
            options_dict = json.load(opt_file)
        pretrain_batch_size = options_dict['batch']

    except:
        pretrain_batch_size =args.batch  #Assume current batch was used in pretraining
        
    #Fine Last Epoch file
    print('load previous state: {0}'.format(trainResultName))
    model.load_state_dict(torch.load(trainResultName, map_location=lambda storage, loc: storage),strict=False)

    #Checking optimizer file
    trainResultName = trainResultName+'o'
    if os.path.exists(trainResultName):
        print('load previous state: {0}'.format(trainResultName))
        optimizer.load_state_dict(torch.load(trainResultName, map_location=lambda storage, loc: storage))

    model.train()
    model.eval()

if bLog:
    logger = Logger(checkpointFolder+'/logs')


""" Save Option Info """
option_str, options_dict = print_options(parser,args)

# option_str += '\nDBList: \n'
# for i, dbname in enumerate(dblist):
#     option_str +=  '{0}:  {1}\n'.format(dbname,X.shape)
# option_str += 'All: {0}'.format(X.shape)
# options_dict['dblist']= dblist

save_options(checkpointFolder, option_str, options_dict)


""" Compute mean and std """
train_X = np.swapaxes(train_X, 1, 2).astype(np.float32) #(num, 73, 240)
train_Y = train_Y.astype(np.float32)

test_X = np.swapaxes(test_X, 1, 2).astype(np.float32) #(num, 73, 240)
test_Y = test_Y.astype(np.float32)

feet = np.array([12,13,14,15,16,17,24,25,26,27,28,29])

Xmean = train_X.mean(axis=2).mean(axis=0)[np.newaxis,:,np.newaxis]
Xmean[:,-7:-4] = 0.0
Xmean[:,-4:]   = 0.5

Xstd = np.array([[[train_X.std()]]]).repeat(train_X.shape[1], axis=1)
Xstd[:,feet]  = 0.9 * Xstd[:,feet]
Xstd[:,-7:-5] = 0.9 * train_X[:,-7:-5].std()
Xstd[:,-5:-4] = 0.9 * train_X[:,-5:-4].std()
Xstd[:,-4:]   = 0.5

""" Data standardization """
train_X = (train_X - Xmean) / Xstd
test_X = (test_X - Xmean) / Xstd

"""Data Shuffle"""
I = np.arange(len(train_X))
rng.shuffle(I); 
train_X = train_X[I]
train_Y = train_Y[I]

print('Input data size: {0}'.format(train_X.shape))

np.savez_compressed(checkpointFolder+'/preprocess_core.npz', Xmean=Xmean, Xstd=Xstd)

#stepNum =0

checkpointFolder_base = os.path.basename(checkpointFolder) 


if train_X.shape[0]  < batch_size:
    batch_size = train_X.shape[0]

curBestAccu = 0
#Compute stepNum start point (to be continuos in tensorboard if pretraine data is loaded)
filelog_str = ''
stepNum = pretrain_epoch* len(np.arange(train_X.shape[0] // pretrain_batch_size))



#LSTM expected (stepSize, batch, features)
train_X
for epoch in range(num_epochs):
    
    #targetFrame = random.choice(range(240))
    model.train()
    batchinds = np.arange(train_X.shape[0] // batch_size)
    rng.shuffle(batchinds)
    
    avgLoss =0
    for bii, bi in enumerate(batchinds):

        idxStart  = bi*batch_size
        # inputDataAll = train_X[idxStart:(idxStart+batch_size),:,targetFrame]      
        # outputDataAll = train_Y[idxStart:(idxStart+batch_size),targetFrame]      
        #outputDataAll = outputDataAll[:,np.newaxis]
        
        inputDataAll = train_X[idxStart:(idxStart+batch_size),:,:]      

        #Reordering from (batchsize,featureNum,frames) ->(frames, batch, features)
        inputDataAll = np.swapaxes(inputDataAll, 0, 1) #(featureNum,batchsize,frames)
        inputDataAll = np.swapaxes(inputDataAll, 0, 2) #(frames,batchsize,featureNum)

        outputDataAll = train_Y[idxStart:(idxStart+batch_size),:]      
        outputDataAll = np.swapaxes(outputDataAll,0,1) #(batch,frames) -> (frame,batch)
        outputDataAll = outputDataAll[:,:,np.newaxis]
        inputData = Variable(torch.from_numpy(inputDataAll)).cuda()
        outputGT = Variable(torch.from_numpy(outputDataAll)).cuda()
    
    
        # ===================forward=====================
        model.hidden = model.init_hidden() #clear out hidden state of the LSTM
        
        output = model(inputData)
        loss = criterion(output, outputGT)

        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================log========================
        #print('model: {}, epoch [{}/{}], loss:{:.4f}'
        #            .format(checkpointFolder_base, epoch +pretrain_epoch, num_epochs, loss.item()))
        avgLoss += loss.item()*batch_size
        
        if bLog:
            # 1. Log scalar values (scalar summary)
            if int(torch.__version__[2])==2:
                info = { 'loss': loss.data[0] }
            else:
                info = { 'loss': loss.item() }

            for tag, value in info.items():
                logger.scalar_summary(tag, value, stepNum)
        
        stepNum = stepNum+1

        # # 2. Log values and gradients of the parameters (histogram summary)
        # for tag, value in model.named_parameters():
        #     tag = tag.replace('.', '/')
        #     logger.histo_summary(tag, value.data.cpu().numpy(), epoch+1)
        #     logger.histo_summary(tag+'/grad', value.grad.data.cpu().numpy(), epoch+1)
    

    temp_str = '## model: {}, epoch [{}/{}], avg loss:{:.4f}\n'.format(checkpointFolder_base, epoch +pretrain_epoch, num_epochs, avgLoss/ (len(batchinds)*batch_size) )


    """compute testing error"""
    batch_size_test = 64
    test_loss = 0.0
    acc= 0.0
    cnt =0.0

    model.eval()
    batchinds = np.arange(test_X.shape[0] // batch_size_test)
    for bii, bi in enumerate(batchinds):

        idxStart  = bi*batch_size_test
        # inputDataAll = test_X[idxStart:(idxStart+batch_size_test),:,-1]      
        inputDataAll = train_X[idxStart:(idxStart+batch_size),:,:]      

        #Reordering from (batchsize,featureNum,frames) ->(frames, batch, features)
        inputDataAll = np.swapaxes(inputDataAll, 0, 1) #(featureNum,batchsize,frames)
        inputDataAll = np.swapaxes(inputDataAll, 0, 2) #(frames,batchsize,featureNum)

        outputDataAll = train_Y[idxStart:(idxStart+batch_size),:]      
        outputDataAll = np.swapaxes(outputDataAll,0,1) #(batch,frames) -> (frame,batch)
        outputDataAll = outputDataAll[:,:,np.newaxis]
        inputData = Variable(torch.from_numpy(inputDataAll)).cuda()
        outputGT = Variable(torch.from_numpy(outputDataAll)).cuda()
    
        model.hidden = model.init_hidden() #clear out hidden state of the LSTM
        output = model(inputData)
        loss = criterion(output, outputGT)
        test_loss += loss.data.cpu().numpy().item()* batch_size_test # sum up batch loss

        #consider all 30 frames in the chuk
        # pred = output.data.cpu().numpy() >= 0.5
        # truth = outputGT.data.cpu().numpy() >= 0.5
        # acc += (pred==truth).sum() /float(train_Y.shape[1]) #train_Y.shape[1] is number of frames (chunk size)

        #consider the last frame
        pred = output.data.cpu().numpy() >= 0.5
        truth = outputGT.data.cpu().numpy() >= 0.5
        acc += (pred[-1,:]==truth[-1,:]).sum()
        cnt += batch_size


    test_loss /= len(batchinds)*batch_size_test
    curBestAccu = max(curBestAccu, acc/cnt)
    print('Testing: modelName: {} (epoch:{}) /Average loss: {:.4f}/Accuracy: {:.4f} (best {:.4f})\n'.format(checkpointFolder_base, epoch +pretrain_epoch,test_loss, acc/cnt,curBestAccu))
    
    print(temp_str)
    filelog_str +=temp_str
    if (epoch + pretrain_epoch) % args.checkpoint_freq == 0:
    #if (epoch + pretrain_epoch) % 1 == 0:
        fileName = checkpointFolder+ '/checkpoint_e' + str(epoch + pretrain_epoch) + '.pth'
        torch.save(model.state_dict(), fileName)
        fileName = checkpointFolder+ '/opt_state.pth'    #overwrite
        torch.save(optimizer.state_dict(), fileName)
        #torch.save(model, fileName)

        file_name = os.path.join(checkpointFolder, 'log.txt')
        with open(file_name, 'a') as opt_file:
            opt_file.write(filelog_str)
            opt_file.write('\n')
            opt_file.close()
        filelog_str =''

