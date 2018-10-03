import os
import sys
import numpy as np
import scipy.io as io
import random


#by jhugestar
sys.path.append('/ssd/codes/glvis_python/')
from Visualize_human_gl import showSkeleton,setSpeech,setSpeechGT,show_Holden_Data_73 #opengl visualization 


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

    # def init_hidden(self):
    #     return (Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_dim)),
    #             Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_dim)))
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
        #lstm_out:  (batch, timesteps, hidden_dim)
        #self.hidden (tuple with two elements):  ( (1, batch, hidden_dim),  (1, batch, hidden_dim))
        lstm_out, self.hidden = self.lstm(
                    input_, self.hidden)
        
        proj = self.proj(lstm_out) #input:(batch, inputDim, outputDim ) -> output (batch, timesteps,1)
        return self.out_act(proj)
#model = naive_lstm2(batch_size).cuda()


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
parser.add_argument('--epochs', type=int, default=50001, metavar='N',
                    help='number of epochs to train (default: 50001)')

parser.add_argument('--batch', type=int, default=512, metavar='N',
                    help='batch size (default: 3072)')

parser.add_argument('--gpu', type=int, default=0, metavar='N',
                    help='Select gpu (default: 0)')

parser.add_argument('--checkpoint_freq', type=int, default=50, metavar='N',
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

# torch.cuda.set_device(args.gpu)

# rng = np.random.RandomState(23456)
# torch.manual_seed(23456)
# torch.cuda.manual_seed(23456)

"""All available dataset"""
#datapath ='/ssd/codes/pytorch_motionSynth/motionsynth_data' 
datapath ='../../motionsynth_data/data/processed/' 

#train_dblist = ['data_panoptic_speech_haggling_sellers_training_byFrame']
# test_dblist = ['data_panoptic_speech_haggling_sellers_testing_byFrame']
#train_dblist = ['data_panoptic_speech_haggling_sellers_training_byFrame_white_tiny']
#test_dblist = ['data_panoptic_speech_haggling_sellers_testing_byFrame_white_tiny']
#test_dblist = ['data_panoptic_speech_haggling_sellers_testing_byFrame_white_tiny']
#test_dblist = ['data_panoptic_speech_haggling_sellers_testing_byFrame_white']
#test_dblist = ['data_panoptic_speech_haggling_sellers_testing_byFrame_white_30frm_tiny']
#test_dblist = ['data_panoptic_speech_haggling_sellers_training_byFrame_white_30frm_tiny']
#test_dblist = ['data_panoptic_speech_haggling_sellers_testing_byFrame_white_30frm']
test_dblist = ['data_panoptic_speech_haggling_sellers_testing_byFrame_white_30frm']

#test_dblist = ['data_panoptic_speech_haggling_sellers_training_byFrame_white_tiny']

test_data = np.load(datapath + test_dblist[0] + '.npz')
test_X= test_data['clips']  #Input (1044,240,73)
test_Y = test_data['classes']  #Input (1044,240,73)


### Data preprocessing ###
checkpointRoot = '/ssd/codes/pytorch_motionSynth/motionsynth_code/classfication_speak/'
loadEpoch = 237
checkpointFolder = checkpointRoot+ 'social_naive_lstm2_try60/'

preprocess = np.load(checkpointFolder + 'preprocess_core.npz') #preprocess['Xmean' or 'Xstd']: (1, 73,1)

""" Data standardization """
test_X = np.swapaxes(test_X, 1, 2).astype(np.float32) #(num, 73, 240)
test_Y = test_Y.astype(np.float32)
test_X_stdd = (test_X - preprocess['Xmean']) / preprocess['Xstd']

""" Load Model """
#model = naive_baseline()#getattr(modelZoo,options_dict['model'])()
batch_size_test = 512
model = naive_lstm2(batch_size_test).cuda()

trainResultName = checkpointFolder + 'checkpoint_e' + str(loadEpoch) + '.pth'
loaded_state = torch.load(trainResultName, map_location=lambda storage, loc: storage)

model.load_state_dict(loaded_state,strict=False) #strict False is needed some version issue for batchnorm
model = model.eval()

batchinds = np.arange(test_X.shape[0] // batch_size_test)
pred_all = np.empty([0,1],dtype=float)
for _, bi in enumerate(batchinds):

    idxStart  = bi*batch_size_test
    inputDataAll = test_X_stdd[idxStart:(idxStart+batch_size_test),:,:]      

    #Reordering from (batchsize,featureNum,frames) ->(batch, frame,features)
    inputDataAll = np.swapaxes(inputDataAll, 1, 2) #(batch, frame,features)
    outputDataAll = test_Y[idxStart:(idxStart+batch_size_test),:]      
    outputDataAll = outputDataAll[:,:,np.newaxis]
        
    inputData = Variable(torch.from_numpy(inputDataAll)).cuda()
    outputGT = Variable(torch.from_numpy(outputDataAll)).cuda()

    model.hidden = model.init_hidden() #clear out hidden state of the LSTM  
    output = model(inputData)
    pred = output.data.cpu().numpy()
    pred_all = np.concatenate((pred_all, pred[:,-1]), axis=0)
    # loss = criterion(output, outputGT)
    # test_loss += loss.data.cpu().numpy().item()* batch_size_test # sum up batch loss

    #     pred = output.data.cpu().numpy() >= 0.5
    #     truth = outputGT.data.cpu().numpy() >= 0.5
    #     acc += (pred==truth).sum() 
    #     cnt += truth.shape[0]

#read body
pred_binary = pred_all[:] >=0.5
pred_binary = pred_binary[:,-1]
from sklearn.metrics import accuracy_score
test_Y_cropped = test_Y[:len(pred_binary),-1]
#acc = accuracy_score(test_Y_, pred_binary)

t = (test_Y_cropped == pred_binary)
correct_samples = sum(t)
acc = float(correct_samples)/len(test_Y_cropped)
print('accuracy: {0:.2f}% (={1}/{2})'.format(acc*100.0,correct_samples,len(test_Y_cropped)))
# #Plot 
# # add a subplot with no frame
# import matplotlib.pyplot as plt
# plt.subplot(221)
# ax2=plt.subplot(311)
# plt.plot(test_Y)
# plt.title('Speech GT')
# ax2=plt.subplot(312)
# plt.plot(pred_binary)
# plt.title('Prediction (binary)')
# ax2=plt.subplot(313)
# plt.plot(pred_all)
# plt.title('Prediction (probability)')
# #plt.ion()
# plt.show()
# #plt.pause(1)

setSpeechGT([test_Y_cropped])
setSpeech([pred_binary])
test_X = np.swapaxes(test_X, 0, 2) #(num, 73, 30) ->(30, 73, num) where num can be thought as frames
show_Holden_Data_73([test_X[-1,:,:]])