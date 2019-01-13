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
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score

#by jhugestar
sys.path.append('/ssd/codes/glvis_python/')
import glViewer


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
# Load Dataset 
#datapath ='/ssd/codes/pytorch_motionSynth/motionsynth_data' 
datapath ='../../motionsynth_data/data/processed/' 
test_dblist_body = ['data_hagglingSellers_speech_body_face_bySequence_white_noGa_brl_testing_tiny']
test_dblist_body = ['data_hagglingSellers_speech_body_face_bySequence_white_noGa_brl_testing']


"""Load body motion data"""
pkl_file_body = open(datapath + test_dblist_body[0] + '.pkl', 'rb')
test_data_body = pickle.load(pkl_file_body)
pkl_file_body.close()

test_body_raw_all = test_data_body['data']  #Input (1044,240,73)
test_seqNames = test_data_body['seqNames']
test_body_initInfo = test_data_body['initInfo']
test_face_raw_all = test_data_body['face']  #Input (1044,240,73)
test_speech_raw_all = test_data_body['speech']      #test_speech_raw_all[seqIdx][pIdx]['indicator']


#######################################
## Set Option
bOwnBody = False
bFaceOnly = False
bBodyOnly = False
#Winner
#Loser
#Buyer
#Seller

#######################################
## Load Network
if bBodyOnly:
    from network_X2speak import Network_body2speak
    net_X2speak = Network_body2speak(bOwnBody)
elif bFaceOnly:
    from network_X2speak import Network_face2speak
    net_X2speak = Network_face2speak(bOwnBody)
else:
    from network_X2speak import Network_facebody2speak
    net_X2speak = Network_facebody2speak(bOwnBody)

#Crop and align the ``last dim" of input_list
def EnsureLastDimLength(input_list):
    #print(input_list)
    length = min([x.shape[-1] for x in input_list])
    #print("EnsureLength: minLength {}".format(length))

    for i in range(len(input_list)):
        dimNum = len(input_list[i].shape)
        if dimNum==1:
            input_list[i] = input_list[i][:length]
        elif dimNum==2:
            input_list[i] = input_list[i][:,:length]
        elif dimNum==3:
            input_list[i] = input_list[i][:,:,:length]

    lengthCheck = max([len(x) for x in input_list])
    #print("EnsureLength: maxLength {}".format(lengthCheck))
    return input_list

def EnsureFirstDimLength(input_list):
    #print(input_list)
    length = min([x.shape[0] for x in input_list])
    #print("EnsureLength: minLength {}".format(length))

    for i in range(len(input_list)):
        dimNum = len(input_list[i].shape)
        if dimNum==1:
            input_list[i] = input_list[i][:length]
        elif dimNum==2:
            input_list[i] = input_list[i][:length,:]
        elif dimNum==3:
            input_list[i] = input_list[i][:length,:,:]

    length = max([x.shape[0] for x in input_list])
    #print("EnsureLength: maxLength {}".format(lengthCheck))
    return input_list

import matplotlib.pyplot as plt
def Plot(input_):
    plt.plot(input_)
    plt.show()



############################
## Mask setting
maskings =[]
maskings += [[0],[1],[2],[3],[4]] #Face Part

for i in range(23):
    maskings.append([3*i+5, 3*i+1+5, 3*i+2+5])
maskings.append( [ 69+5, 70+5, 71+5, 72+5]) #Foot part
maskings.append( range(5)) #All face part
maskings.append( range(5,78)) #All body part
maskings.append( []) #No Masking

for maskIdx in range(len(maskings)):
    posErr_list = []
    faceOriErr_list = []
    bodyOriErr_list = []
    traj_list_seq =[]
    bVisualize = False
    accuracy_sum = 0
    cnt_sum=0
    for seqIdx in range(len(test_face_raw_all)):

        seqName_base = os.path.basename(test_seqNames[seqIdx])
        # if bVisualize == True and not ('170228_haggling_b2_group1' in seqName_base):
        #     continue

        #print('{}'.format(seqName_base))
        
        face_group = test_face_raw_all[seqIdx]     #(1, frames, feature:73)
        body_group = test_body_raw_all[seqIdx]     #(1, frames, feature:73)
        speech_group = test_speech_raw_all[seqIdx]     #(1, frames, feature:73)

        # """Own Body"""
        if bOwnBody:
            test_face = face_group[1,:,:]       #(frames, dim:5)
            test_body = body_group[1,:,:]       #(frames, dim:73)
            [test_face, test_body] =  EnsureFirstDimLength([test_face, test_body])
            test_X = np.concatenate( (test_face, test_body),axis=1)     #(frames, dim:78)
            test_Y = np.array(speech_group[1]['indicator'])

            #making them as (batch, featureDim, freams)
            test_X = np.expand_dims(test_X,0)   #(1, freams, dim)
            test_X = np.swapaxes(test_X, 1, 2).astype(np.float32) #(num, featureDim, frames)
            test_Y = test_Y.astype(np.float32)

            [test_X, test_Y] =  EnsureLastDimLength([test_X, test_Y])

        else:       #Other Subject Body (social)
            test_face = face_group[2,:,:]       #(frames, dim:5)
            test_body = body_group[2,:,:]       #(frames, dim:73)
            [test_face, test_body] =  EnsureFirstDimLength([test_face, test_body])
            test_X = np.concatenate( (test_face, test_body),axis=1)     #(frames, dim:78)
            test_Y = np.array(speech_group[1]['indicator'])

            #making them as (batch, featureDim, freams)
            test_X = np.expand_dims(test_X,0)   #(1, freams, dim)
            test_X = np.swapaxes(test_X, 1, 2).astype(np.float32) #(num, featureDim, frames)
            test_Y = test_Y.astype(np.float32)

            [test_X, test_Y] =  EnsureLastDimLength([test_X, test_Y])


        """Other Subject Body by shuffling (non-social)"""
        # #Select random sequence
        # idxCand = range(len(test_X_raw_all))
        # idxCand.remove(seqIdx)
        # rng.shuffle(idxCand)
        # randomIdx = idxCand[0]
        # print("randomIdx: {} vs ({})".format(randomIdx,seqIdx))
        # test_sppech_raw = test_Y_raw_all[randomIdx]     #(3, frames)

        # #Adjust length
        # frameLeng = min(test_sppech_raw[0]['indicator'].shape[0], test_X_raw.shape[1])
        # test_X = test_X_raw[1,:frameLeng,:]      #(num, 73) 
        # frameLeng = test_X.shape[0]
        # test_X = np.concatenate( (test_X, test_X_raw[2,:frameLeng,:]), axis= 0)      #(num, 73)

        # test_Y = test_sppech_raw[2]['indicator']
        # test_Y = test_Y[:frameLeng]
        # test_Y2 = test_sppech_raw[1]['indicator']
        # test_Y2 = test_Y2[:frameLeng]
        # test_Y = np.concatenate( (test_Y, test_Y2), axis= 0)

        
        ######################################
        # Standardization
        test_X_std = net_X2speak.standardize_input(test_X)  #(1, features:10, frames)


        ######################################
        # Mask out some part
        for i in maskings[maskIdx]:
            test_X_std[0,i,:] *=0

        if bBodyOnly:
            test_X_std[0,:5,:]*=0
        if bFaceOnly:
            test_X_std[0,5:,:]*=0

        #######################################################
        # Predicting face by Face2Face
        inputData_ = Variable(torch.from_numpy(test_X_std)).cuda()  #(batch, 3, frameNum)
        outputData_ = net_X2speak(inputData_)
        
        ######################################
        # De-Standardization (no need)
        outputData_ = outputData_.data.cpu().numpy()  
        pred_speak = np.squeeze(outputData_)  #(frames,)

        # ####################################
        # ## Compute Errors
        pred_speak_binary = (pred_speak>0.5)*1.0
        accuracy = accuracy_score(test_Y,pred_speak_binary)
        
        #print("accuracy: {}".format(float(accuracy)))

        # if True:
        #     import matplotlib.pyplot as plt
        #     plt.subplot(2,1,1)
        #     plt.plot(test_Y)
        #     plt.hold(True)
        #     plt.subplot(2,1,2)
        #     plt.plot(pred_speak_binary)
        #     plt.hold(True)
        #     plt.plot(pred_speak)
        #     plt.hold(True)
        #     plt.show()
        #     plt.pause(0.5)

        accuracy_sum += accuracy* len(pred_speak)
        cnt_sum +=len(pred_speak)
        #accuracy = float(correct.item()) / len(outputData_np)
        #accuracy = sum(pred>0.5) == test_Y) / len(pred)
        #print("accuracy: {}".format(float(accuracy)/len(pred_speak)))

        # """ Plot output"""
        # import matplotlib.pyplot as plt
        # plt.plot(np.squeeze(pred))
        # plt.hold(True)
        # plt.plot(outputData_np)
        # plt.show()


        #continue
        
        # traj_list_seq.append(np.array(traj_list))

    acc_avg = float(accuracy_sum)/cnt_sum

    #print("maskIdx: {}, acc_avg: {}".format(maskIdx-5,acc_avg))
    print("{}".format(acc_avg))
    #print("acc_avg: {}".format(acc_avg))

