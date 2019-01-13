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
import utility
from utility import print_options,save_options
from utility import setCheckPointFolder
from utility import my_args_parser
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score
from sklearn.metrics import hamming_loss

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
# To visualize body 
sys.path.append('/ssd/codes/pytorch_motionSynth/motionsynth_data/motion')
#from Pivots import Pivots
from Quaternions import Quaternions

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




"""Load formation data"""
test_dblist = ['data_hagglingSellers_speech_formation_bySequence_white_brl_testing_4fcn']   #no normalized
pkl_file = open(datapath + test_dblist[0] + '.pkl', 'rb')
test_data = pickle.load(pkl_file)
pkl_file.close()

test_traj_raw_all = test_data['data']  #Input (1044,240,73)
#speech_raw_all = test_data['speech']  #Input (1044,240,73)
test_seqNames = test_data['seqNames']


#######################################
## Set Option
# iters = [[True, True, True], [True, True, False],[True, False, True],[False, True, True],[False, False, True],[False, True, False], [True, False, False] , [False, False, False]]
bOwnBody = False
bFaceOnly = True
bBodyOnly = False

# for it in iters:
#     bOwnBody =it[0]
#     bFaceOnly =it[1]
#     bBodyOnly =it[2]
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
## Choose a sequence
#seqIdx =1
posErr_list = []
faceOriErr_list = []
bodyOriErr_list = []
traj_list_seq =[]
bVisualize = True
accuracy_sum = 0
cnt_sum=0
for seqIdx in range(len(test_face_raw_all)):

    seqName_base = os.path.basename(test_seqNames[seqIdx])
    # if bVisualize == True and not ('170228_haggling_b1_group6' in seqName_base):
    #     continue

    if bVisualize == True and not ('170221_haggling_b3_group1' in seqName_base):
        continue


    #print('{}'.format(seqName_base))
    
    face_group = test_face_raw_all[seqIdx]     #(1, frames, feature:73)
    body_group = test_body_raw_all[seqIdx]     #(1, frames, feature:73)
    speech_group = test_speech_raw_all[seqIdx]     #(1, frames, feature:73)


    #Compute turn-taking goodness?
    seller1_speak = np.array(speech_group[1]['indicator'])*1.0
    seller2_speak = np.array(speech_group[2]['indicator'])*1.0
    speaksum = seller1_speak+seller2_speak
    turn_taking_goodness = (speaksum<=1)*1.0
    turn_taking_goodness = sum(turn_taking_goodness)/ len(turn_taking_goodness) 
    #print("turn_taking_goodness: {}".format(turn_taking_goodness))
    print("{}".format(turn_taking_goodness))



    # """Own Body"""
    if bOwnBody:
        test_face = face_group[2,:,:]       #(frames, dim:5)
        test_body = body_group[2,:,:]       #(frames, dim:73)
        [test_face, test_body] =  EnsureFirstDimLength([test_face, test_body])
        test_X = np.concatenate( (test_face, test_body),axis=1)     #(frames, dim:78)
        test_Y = np.array(speech_group[2]['indicator'])

        #making them as (batch, featureDim, freams)
        test_X = np.expand_dims(test_X,0)   #(1, freams, dim)
        test_X = np.swapaxes(test_X, 1, 2).astype(np.float32) #(num, featureDim, frames)
        test_Y = test_Y.astype(np.float32)

        [test_X, test_Y] =  EnsureLastDimLength([test_X, test_Y])

    else:       #Other Subject Body (social)
        test_face = face_group[1,:,:]       #(frames, dim:5)
        test_body = body_group[1,:,:]       #(frames, dim:73)
        [test_face, test_body] =  EnsureFirstDimLength([test_face, test_body])
        test_X = np.concatenate( (test_face, test_body),axis=1)     #(frames, dim:78)
        test_Y = np.array(speech_group[2]['indicator'])

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
    test_X_std = net_X2speak.standardize_input(test_X)  #(1, features, frames)

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
    correctCase = (test_Y==pred_speak_binary)*1.0
    
    #print("seq: {}, accuracy: {}".format(seqName_base, float(accuracy)))
    #print("seq: {}".format(seqName_base))
    #print("{}".format(float(accuracy)))

    if False:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        fig.set_size_inches(18.5, 10.5)
        plt.subplot(3,1,1)
        plt.title('GT', fontsize=15)
        plt.plot(test_Y)
        plt.hold(True)
        plt.subplot(3,1,2)
        plt.title('Prediction', fontsize=15)
        plt.plot(pred_speak_binary)
        plt.hold(True)
        plt.plot(pred_speak)
        plt.hold(True)
        plt.subplot(3,1,3)

        plt.title('Accuracy:{0:.2f}'.format(accuracy*100.0), fontsize=15)
        plt.plot(correctCase)
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.5)


        """Save the graph as files"""
        # plt.close()
        # if bOwnBody:
        #     if bBodyOnly==False and bFaceOnly==False:
        #         fileName='{}-OwnSig-all'.format(seqName_base)
        #     elif bBodyOnly==True and bFaceOnly==False:
        #         fileName='{}-OwnSig-body'.format(seqName_base)
        #     else:
        #         fileName='{}-OwnSig-face'.format(seqName_base)
        # else:
        #     if bBodyOnly==False and bFaceOnly==False:
        #         fileName='{}-OtherSig-all'.format(seqName_base)
        #     elif bBodyOnly==True and bFaceOnly==False:
        #         fileName='{}-OtherSig-body'.format(seqName_base)
        #     else:
        #         fileName='{}-OtherSig-face'.format(seqName_base)
        # fig.savefig('/ssd/thesis_fig/speaking/'+fileName+'.png')

    accuracy_sum += accuracy* len(pred_speak)
    cnt_sum +=len(pred_speak)
    #accuracy = float(correct.item()) / len(outputData_np)
    #accuracy = sum(pred>0.5) == test_Y) / len(pred)
    #print("accuracy: {}".format(float(accuracy)/len(pred_speak)))

    if bVisualize==False:
        continue

    
    # ####################################
    # ## Visualize Body
    vis_bodyData = [body_group[2,:,:], body_group[0,:,:], body_group[1,:,:]]
    vis_bodyData = [f.swapaxes(0,1) for f in vis_bodyData]
    
    vis_bodyGT_initTrans = [test_body_initInfo[seqIdx][2]['pos'], test_body_initInfo[seqIdx][0]['pos'], test_body_initInfo[seqIdx][1]['pos'] ]
    vis_bodyGT_initRot = [test_body_initInfo[seqIdx][2]['rot'], test_body_initInfo[seqIdx][0]['rot'], test_body_initInfo[seqIdx][1]['rot']]
    vis_bodyGT_initRot = [ Quaternions(x) for x in vis_bodyGT_initRot ]
    vis_body_initRot = [body_group[2,:,:], body_group[0,:,:], body_group[1,:,:]]
    vis_body_initTrans = [body_group[2,:,:], body_group[0,:,:], body_group[1,:,:]]
    """Visualize Body"""
    glViewer.set_Holden_Data_73(vis_bodyData, ignore_root=False, initRot=vis_bodyGT_initRot, initTrans= vis_bodyGT_initTrans, bIsGT=False)


    # # ####################################
    # # ## Visualize Face
    # vis_faceData = [face_group[2,:,:], face_group[0,:,:], face_group[1,:,:]]
    # vis_faceData = [f.swapaxes(0,1) for f in vis_faceData]
    # #glViewer.SetFaceParmData(vis_faceData,True)

    # vis_faceNormalData= [ test_traj_raw_all[seqIdx][2,:,3:6], test_traj_raw_all[seqIdx][0,:,3:6], test_traj_raw_all[seqIdx][1,:,3:6]]
    # vis_faceNormalData = [f.swapaxes(0,1) for f in vis_faceNormalData]
    # vis_f2face_trans , vis_f2face_rot = utility.ComputeHeadOrientation(glViewer.g_skeletons, vis_faceNormalData)
    # glViewer.SetFaceParmDataWithTrans(vis_faceData,True, vis_f2face_trans, vis_f2face_rot)


    # ####################################
    # ## Visualize Speaking
    vis_speakGT = [speech_group[2]['indicator'], speech_group[0]['indicator'], speech_group[1]['indicator']]
    glViewer.setSpeechGT_binary(vis_speakGT)

    vis_speak_pred = [pred_speak_binary]# speech_group[0]['indicator'], speech_group[1]['indicator']]
    glViewer.setSpeech_binary(vis_speak_pred)


    glViewer.init_gl()

    # """Compute turn changing time"""
    # speechSig = test_sppech_raw[2,:]    
    # turnChange = np.where(abs(speechSig[1:] - speechSig[:-1] ) >0.5)[0]
    # """Show only turn change time"""
    # frameLeng = posData[0].shape[1]
    # print(turnChange)
    # selectedFrames = []
    # for f in turnChange:
    #     fStart = max(f - 90,0)
    #     fEnd =  min(f + 90,frameLeng-1)
    #     selectedFrames += range(fStart,fEnd)

    # for i in range(len(posData)):
    #     posData[i] = posData[i][:,selectedFrames]
    #     faceNormalData[i] = faceNormalData[i][:,selectedFrames]
    #     bodyNormalData[i] = bodyNormalData[i][:,selectedFrames]
        

    # glViewer.resetFrameLimit()
    # glViewer.setPosOnly(posData)
    # glViewer.setFaceNormal(faceNormalData)
    # glViewer.setBodyNormal(bodyNormalData)

    
    # traj_list_seq.append(np.array(traj_list))

acc_avg = float(accuracy_sum)/cnt_sum

print("acc_avg: {}".format(acc_avg))



# ##Draw Error Figure
# avg_posErr_list=[]
# total_avg_posErr = 0
# cnt=0
# for p in posErr_list:
#     avgValue = np.mean(p)
#     avg_posErr_list.append(avgValue)
#     print(avgValue)
#     total_avg_posErr += avgValue*len(p)
#     cnt += len(p)

# total_avg_posErr = total_avg_posErr/cnt
# std = np.std(avg_posErr_list)
# print("total_avg_posErr: {}, std {}".format(total_avg_posErr, std))


# ##Draw Error Figure
# avg_bodyOriErr_list=[]
# total_avg_bodyOriErr = 0
# cnt=0
# for p in bodyOriErr_list:
#     avgValue = np.mean(p)
#     avg_bodyOriErr_list.append(avgValue)
#     #print(avgValue)
#     total_avg_bodyOriErr += avgValue*len(p)
#     cnt += len(p)

# total_avg_bodyOriErr = total_avg_bodyOriErr/cnt
# std = np.std(avg_bodyOriErr_list)
# print("total_avg_bodyOriErr: {}, std {}".format(total_avg_bodyOriErr,std))


# ##Draw Error Figure
# avg_faceOriErr_list=[]
# total_avg_faceOriErr = 0
# cnt=0
# for p in faceOriErr_list:
#     avgValue = np.mean(p)
#     avg_faceOriErr_list.append(avgValue)
#     #print(avgValue)
#     total_avg_faceOriErr += avgValue*len(p)
#     cnt += len(p)

# total_avg_faceOriErr = total_avg_faceOriErr/cnt
# std = np.std(avg_faceOriErr_list)
# print("total_avg_faceOriErr: {}, std {}".format(total_avg_faceOriErr,std))


# #save current values
# output = open('predForm_1112_noNorm.pkl', 'wb')
# pickle.dump({'avg_posErr_list':avg_posErr_list, 'avg_bodyOriErr_list':avg_bodyOriErr_list, 'avg_faceOriErr_list': avg_faceOriErr_list}, output)
# output.close()



# # #load current values
# # pkl_file = open('predForm_1112_noNorm.pkl', 'rb')
# # data = pickle.load(pkl_file)
# # pkl_file.close()
# # avg_posErr_list = data['avg_posErr_list']
# # avg_bodyOriErr_list = data['avg_bodyOriErr_list']
# # avg_faceOriErr_list = data['avg_faceOriErr_list']



# # add a subplot with no frame
# bShowGraph = True
# if bShowGraph:
#     import matplotlib.pyplot as plt
#     plt.rc('xtick', labelsize=18)     
#     plt.rc('ytick', labelsize=18)
    

#     ax2=plt.subplot(311)
#     plt.plot(avg_posErr_list)
#     plt.title('Average Pos Error', fontsize=20)
#     plt.grid()
#     plt.xlabel('Seq. Index', fontsize=20)
#     plt.ylabel('Error (cm)', fontsize=20)
    

#     ax2=plt.subplot(312)
#     plt.plot(avg_bodyOriErr_list)
#     plt.title('Average body Orientation Error', fontsize=20)
#     plt.grid()
#     #plt.xlabel('Seq. Index', fontsize=15)
#     plt.ylabel('Error (cm)', fontsize=20)

#     ax2=plt.subplot(313)
#     plt.plot(avg_faceOriErr_list)
#     plt.title('Average face Orientation Error', fontsize=20)
#     plt.grid()
#     #plt.xlabel('Seq. Index', fontsize=15)
#     plt.ylabel('Error (cm)', fontsize=20)
    

#     plt.tight_layout()
#     plt.show()




