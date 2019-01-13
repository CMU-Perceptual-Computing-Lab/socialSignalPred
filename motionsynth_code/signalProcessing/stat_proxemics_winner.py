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

#import modelZoo

import cPickle as pickle

# Utility Functions
import utility
from utility import print_options,save_options
from utility import setCheckPointFolder
from utility import my_args_parser

from utility import data_2dTo3D
from utility import ConvertTrajectory_velocityForm

from sklearn.preprocessing import normalize

sys.path.append('../../motionsynth_data/motion')
#import BVH as BVH
#import Animation as Animation
from Quaternions import Quaternions
from Pivots import Pivots

import matplotlib.pyplot as plt




#by jhugestar
sys.path.append('/ssd/codes/glvis_python/')
#from glViewer import SetFaceParmData,setSpeech,setSpeechGT,setSpeech_binary, setSpeechGT_binary, init_gl #opengl visualization 
import glViewer

sys.path.append('/ssd/codes/pytorch_motionSynth/motionsynth_data/motion')
from Pivots import Pivots


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
# Dataset 
#datapath ='/ssd/codes/pytorch_motionSynth/motionsynth_data' 
datapath ='../../motionsynth_data/data/processed/' 

#test_dblist = ['data_hagglingSellers_speech_face_60frm_10gap_white_testing']
#test_dblist = ['data_hagglingSellers_speech_formation_30frm_10gap_white_testing']

#test_dblist = ['data_hagglingSellers_speech_formation_bySequence_white_testing']
#test_dblist = ['data_hagglingSellers_speech_formation_pNorm_bySequence_white_testing']

#test_dblist = ['data_hagglingSellers_speech_formation_pN_rN_rV_bySequence_white_testing_beta']
#test_dblist = ['data_hagglingSellers_speech_formation_pN_rN_rV_bySequence_white_testing_4fcn']

test_dblist = ['data_hagglingSellers_speech_formation_bySequence_white_brl_all_4fcn_atten']   #no normalized
test_dblist = ['data_hagglingSellers_speech_formation_bySequence_white_brl_all_4fcn_norm']   #no normalized
test_dblist = ['data_hagglingSellers_speech_formation_bySequence_white_bwl_all_4fcn_norm']   #no normalized


#test_dblist = ['data_hagglingSellers_speech_formation_pN_rotS_bySequence_white_training']
pkl_file = open(datapath + test_dblist[0] + '.pkl', 'rb')
test_data = pickle.load(pkl_file)
pkl_file.close()

test_X_raw_all = test_data['data']  #Input (1044,240,73)
test_Y_raw_all = test_data['speech']  #Input (1044,240,73)
test_seqNames = test_data['seqNames']

test_attention_all = test_data['attention'] #face,body 

# test_refPos_all = test_data['refPos'] #to go to the original position
# test_refRot_all = test_data['refRot'] #to go to the original orientation. Should take inverse for the quaternion



# ########################
# ### Compare angles


angle_br_list = []
angle_bl_list = []
angle_rl_list = []

angle_histo_list = None
totalCnt = 0
rightAngleBetterCnt =0
#Compute body normal 2 
for seqIdx in range(len(test_X_raw_all)):

    seqName_base = os.path.basename(test_seqNames[seqIdx])
    # if bVisualize == True and not ('170228_haggling_b2_group1' in seqName_base):
    #     continue

    print('{}-{}'.format(seqName_base, 0))  #(3, 1832, 9)

    test_X_raw = test_X_raw_all[seqIdx]     #(3, frames, feature:9)
    test_attention = test_attention_all[seqIdx]
    test_sppech_raw = test_Y_raw_all[seqIdx]     #(3, frames)

    vect_b2r = test_X_raw[1,:,:3] - test_X_raw[0,:,:3]   #Buyer to Right Seller Vector
    vect_b2r = normalize(vect_b2r,axis=1) #(frames,3)
    vect_z = np.array([[0,0,1]]).repeat(len(vect_b2r), axis=0)
    rotation = Quaternions.between(vect_z, vect_b2r)[:,np.newaxis]    
    angleValue_z2r = abs(Pivots.from_quaternions(rotation).ps)

    #utility.Plot(angleValue_z2r)



    vect_b2l = test_X_raw[2,:,:3] - test_X_raw[0,:,:3]   #Buyer to Right Seller Vector
    vect_b2l = normalize(vect_b2l,axis=1) #(frames,3)
    vect_z = np.array([[0,0,1]]).repeat(len(vect_b2l), axis=0)
    rotation = Quaternions.between(vect_z, vect_b2l)[:,np.newaxis]    
    angleValue_z2l = abs(Pivots.from_quaternions(rotation).ps)

    #utility.Plot(angleValue_z2r)


    
    angle_br = np.mean(angleValue_z2r)   #Vector 2 Angles
    angle_br_list.append(angle_br)

    angle_bl = np.mean(angleValue_z2l)
    angle_bl_list.append(angle_bl)

    angleRatio = angleValue_z2r / (angleValue_z2r+angleValue_z2l)   #0 means winner direction, 1 means loser direction.
    if angle_histo_list is None:
        angle_histo_list = angleRatio
    else:
        angle_histo_list = np.concatenate((angle_histo_list,angleRatio),axis=0)

    totalCnt +=1
    if(angle_br < angle_bl):
        rightAngleBetterCnt+=1

import matplotlib.pyplot as plt
import numpy as np
fig = plt.figure()
ax = fig.add_subplot(1, 1.1, 1)
major_ticks = np.arange(0, 1, 0.1)
ax.set_xticks(major_ticks)

major_ticks = np.arange(0, 5000, 500)
ax.set_yticks(major_ticks)
ax.set_ylim([0,5000])
plt.hist(angle_histo_list, normed=False, bins=100,histtype='bar',color = 'red',edgecolor="k",linewidth=1)
plt.rcParams["patch.force_edgecolor"] = True
plt.grid(True)
plt.show()


# import pylab as P

# # the histogram of the data with histtype='step'
# n, bins, patches = P.hist(angle_histo_list, 100, normed=1, histtype='bar',rwidth=0.8)
# P.setp(patches, 'facecolor', 'g', 'alpha', 0.75)
# P.show()

# # add a line showing the expected distribution
# l = P.plot(bins, y, 'k--', linewidth=1.5)


# #
# # create a histogram by providing the bin edges (unequally spaced)
# #
# P.figure()

# bins = np.array(range(0,101))/100.0
# # the histogram of the data with histtype='step'
# n, bins, patches = P.hist(angle_histo_list, bins, normed=1, histtype='bar', rwidth=0.8)
# P.show()
#


print('winner is close in {} percent'.format(rightAngleBetterCnt/float(totalCnt)))
    
print("b-r{} ({})/{}/{}\n b-l{} ({})/{}/{}\n".format(np.mean(angle_br_list), np.std(angle_br_list), np.min(angle_br_list), np.max(angle_br_list),
                                                                np.mean(angle_bl_list), np.std(angle_bl_list), np.min(angle_bl_list), np.max(angle_bl_list)))



# dist_br_list = []
# dist_bl_list = []
# dist_rl_list = []

dist_bw_list = []
dist_bl_list = []
dist_wl_list = []

medianShape_list = []
totalCnt = 0
winnerCloserCnt =0
for seqIdx in range(len(test_X_raw_all)):

    seqName_base = os.path.basename(test_seqNames[seqIdx])
    # if bVisualize == True and not ('170228_haggling_b2_group1' in seqName_base):
    #     continue

    print('{}-{}'.format(seqName_base, 0))  #(3, 1832, 9)

    test_X_raw = test_X_raw_all[seqIdx]     #(3, frames, feature:9)
    test_attention = test_attention_all[seqIdx]
    test_sppech_raw = test_Y_raw_all[seqIdx]     #(3, frames)

    dist_bw = test_X_raw[0,:,:3] - test_X_raw[1,:,:3]
    dist_bw = np.mean(np.sqrt(np.sum(dist_bw**2,axis=1)))
    dist_bw_list.append(dist_bw)

    dist_bl = test_X_raw[0,:,:3] - test_X_raw[2,:,:3]
    dist_bl = np.mean(np.sqrt(np.sum(dist_bl**2,axis=1)))
    dist_bl_list.append(dist_bl)

    dist_wl = test_X_raw[1,:,:3] - test_X_raw[2,:,:3]
    dist_wl = np.mean(np.sqrt(np.sum(dist_wl**2,axis=1)))
    dist_wl_list.append(dist_wl)

    medianFrame = int(test_X_raw.shape[1]/2)
    medianShape = test_X_raw[:,medianFrame,:3]
    medianShape_list.append(medianShape)

    totalCnt +=1

    if(dist_bw < dist_bl):
        winnerCloserCnt+=1


print('winner is close in {} percent'.format(winnerCloserCnt/float(totalCnt)))
    

dist_bw_avg =np.mean(dist_bw_list)
dist_bl_avg = np.mean(dist_bl_list)
dist_wl_avg = np.mean(dist_wl_list)

print("b-w{} ({})/{}/{}\n b-l{} ({})/{}/{}\n w-l{} ({})/{}/{}".format(np.mean(dist_bw_list), np.std(dist_bw_list), np.min(dist_bw_list), np.max(dist_bw_list),
                                                                np.mean(dist_bl_list), np.std(dist_bl_list), np.min(dist_bl_list), np.max(dist_bl_list),
                                                                np.mean(dist_wl_list), np.std(dist_wl_list), np.min(dist_wl_list), np.max(dist_wl_list)))



# ########################
# ### Draw polygon
# bDrawPolygon = True
# if bDrawPolygon:
#     from matplotlib.patches import Polygon
#     import matplotlib
#     from matplotlib.collections import PatchCollection

#     fig, ax = plt.subplots()
#     patches = []
#     num_polygons = 5
#     num_sides = 3

#     for s in medianShape_list:   #s: (3,3)
#         points  = s[:,(0,2)]
#         polygon = Polygon(points, True)       #input Nx2
#         patches.append(polygon)


#     p = PatchCollection(patches, cmap=matplotlib.cm.jet, alpha=0.1)

#     colors = 100*np.random.rand(len(patches))
#     p.set_array(np.array(colors))

#     ax.add_collection(p)
#     ax.tick_params(axis = 'both', which = 'major', labelsize = 20)

#     plt.ylim((-50,300))
#     plt.xlim((-150,150))
#     plt.xlabel('X-axis (cm)', fontsize=20)
#     plt.ylabel('Z-axis (cm)', fontsize=20)
#     plt.grid()
#     plt.axis('equal')
#     plt.tight_layout()
#     plt.show()



# ############################
# # Draw Proxemics Heat map
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.stats import multivariate_normal

# # create 2 kernels
# # m1 = (50,50)
# # s1 = np.eye(2)*20
# # k1 = multivariate_normal(mean=m1, cov=s1)

# # m2 = (-50,50)
# # s2 = np.eye(2)*20
# # k2 = multivariate_normal(mean=m2, cov=s2)


# # create a grid of (x,y) coordinates at which to evaluate the kernels
# fig, ax = plt.subplots()
# xlim = (-150, 150)
# ylim = (300,-50)#-50, 300)
# # plt.ylim(ylim)
# # plt.xlim(xlim)
# xres = 500
# yres = 500

# x = np.linspace(xlim[0], xlim[1], xres)
# y = np.linspace(ylim[0], ylim[1], yres)
# xx, yy = np.meshgrid(x,y)
# xxyy = np.c_[xx.ravel(), yy.ravel()]
# zz= None

# for i, s in enumerate(medianShape_list):   #s: (3,3)

#     for id in range(1,3):
#         points  = s[id,(0,2)]
#         m1 = (points[0],points[1])
#         s1 = np.eye(2)*40
#         k1 = multivariate_normal(mean=m1, cov=s1)
#         if zz is None:
#             zz = k1.pdf(xxyy)
#         else:
#             zz = zz+ k1.pdf(xxyy)
    
#     # if i>5:
#     #     break 

# # evaluate kernels at grid points

# # reshape and plot image
# img = zz.reshape((xres,yres))
# plt.imshow(img, extent=[-150,150,-50, 300]); 
# plt.grid()
# ax.tick_params(axis = 'both', which = 'major', labelsize = 20)
# #plt.axis('equal')
# plt.show()













# ############################
# # Error plot
# # add a subplot with no frame
# bShowGraph = False
# if bShowGraph:
#     plt.rc('xtick', labelsize=18)     
#     plt.rc('ytick', labelsize=18)
    

#     ax2=plt.subplot(311)
#     plt.plot(dist_br_list)
#     plt.title('dist_br_list', fontsize=20)
#     plt.grid()
#     plt.xlabel('Seq. Index', fontsize=20)
#     plt.ylabel('Error (cm)', fontsize=20)
    

#     ax2=plt.subplot(312)
#     plt.plot(dist_bl_list)
#     plt.title('dist_bl_list', fontsize=20)
#     plt.grid()
#     #plt.xlabel('Seq. Index', fontsize=15)
#     plt.ylabel('Error (cm)', fontsize=20)

#     ax2=plt.subplot(313)
#     plt.plot(dist_rl_list)
#     plt.title('dist_rl_list', fontsize=20)
#     plt.grid()
#     #plt.xlabel('Seq. Index', fontsize=15)
#     plt.ylabel('Error (cm)', fontsize=20)
    

#     plt.tight_layout()
#     plt.show()



# # #load current values
# # pkl_file = open('predForm_1112_noNorm.pkl', 'rb')
# # data = pickle.load(pkl_file)
# # pkl_file.close()
# # avg_posErr_list = data['avg_posErr_list']
# # avg_bodyOriErr_list = data['avg_bodyOriErr_list']
# # avg_faceOriErr_list = data['avg_faceOriErr_list'] 





