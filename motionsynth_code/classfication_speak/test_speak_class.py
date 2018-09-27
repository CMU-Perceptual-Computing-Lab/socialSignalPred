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

#import modelZoo


class naive_baseline(nn.Module):
    def __init__(self):
        super(naive_baseline, self).__init__()
        self.fc1 = nn.Linear(73, 73)
        self.relu1 = nn.Sequential(nn.PReLU(), 
                        nn.BatchNorm1d(73))
        self.dout = nn.Dropout(0.2)

        self.fc2 = nn.Linear(73, 200)
        self.relu2 = nn.Sequential(nn.PReLU(), 
                        nn.BatchNorm1d(200))
        self.dout = nn.Dropout(0.2)

        self.fc3 = nn.Sequential( nn.Linear(200, 1), 
                        nn.BatchNorm1d(1)) 
        self.out_act = nn.Sigmoid()

    def forward(self, input_):
        a1 = self.fc1(input_)
        h1 = self.relu1(a1)
        h1 = self.dout(h1)

        a2 = self.fc2(h1)
        h2 = self.relu2(a2)
        h2 = self.dout(h2)

        a3 = self.fc3(h2)
        y = self.out_act(a3)
        return y
#model = naive_baseline().cuda()


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

parser.add_argument('--batch', type=int, default=3072, metavar='N',
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
test_dblist = ['data_panoptic_speech_haggling_sellers_testing_byFrame_white']
#test_dblist = ['data_panoptic_speech_haggling_sellers_training_byFrame_white_tiny']

test_data = np.load(datapath + test_dblist[0] + '.npz')
test_X= test_data['clips']  #Input (1044,240,73)
test_Y = test_data['classes']  #Input (1044,240,73)


### Data preprocessing ###
checkpointRoot = '/ssd/codes/pytorch_motionSynth/motionsynth_code/classfication_speak/'
checkpointFolder = checkpointRoot+ 'social_naive_baseline_try34/'

preprocess = np.load(checkpointFolder + 'preprocess_core.npz') #preprocess['Xmean' or 'Xstd']: (1, 73,1)

""" Data standardization """
test_X = np.swapaxes(test_X, 1, 2).astype(np.float32) #(num, 73, 240)
test_Y = test_Y.astype(np.float32)
test_X_stdd = (test_X - preprocess['Xmean']) / preprocess['Xstd']

""" Load Model """
model = naive_baseline()#getattr(modelZoo,options_dict['model'])()
loadEpoch = 2550
trainResultName = checkpointFolder + 'checkpoint_e' + str(loadEpoch) + '.pth'
loaded_state = torch.load(trainResultName, map_location=lambda storage, loc: storage)

model.load_state_dict(loaded_state,strict=False) #strict False is needed some version issue for batchnorm
model = model.eval()

batch_size_test = 512
batchinds = np.arange(test_X.shape[0] // batch_size_test)
pred_all = np.empty([0,1],dtype=float)
for _, bi in enumerate(batchinds):

    idxStart  = bi*batch_size_test
    inputDataAll = test_X_stdd[idxStart:(idxStart+batch_size_test),:,0]      
    outputDataAll = test_Y[idxStart:(idxStart+batch_size_test),:]      
        
    inputData = Variable(torch.from_numpy(inputDataAll))
    outputGT = Variable(torch.from_numpy(outputDataAll))

    output = model(inputData)
    pred = output.data.numpy()
    pred_all = np.concatenate((pred_all, pred), axis=0)
    # loss = criterion(output, outputGT)
    # test_loss += loss.data.cpu().numpy().item()* batch_size_test # sum up batch loss

    #     pred = output.data.cpu().numpy() >= 0.5
    #     truth = outputGT.data.cpu().numpy() >= 0.5
    #     acc += (pred==truth).sum() 
    #     cnt += truth.shape[0]

#read body
pred_binary = pred_all[:] >=0.5
from sklearn.metrics import accuracy_score
test_Y_ = test_Y[:len(pred_binary)]
#acc = accuracy_score(test_Y_, pred_binary)

t = (test_Y_ == pred_binary)
correct_samples = sum(sum(t))
acc = float(correct_samples)/len(test_Y)
print('accuracy: {0:.2f}% (={1}/{2})'.format(acc*100.0,correct_samples,len(test_Y)))
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


setSpeechGT([test_Y_])
setSpeech([pred_binary])
test_X = np.swapaxes(test_X, 0, 2) #(num, 73, 1) ->(1, 73, num) where num can be thought as frames
show_Holden_Data_73([test_X[0,:,:]])