""" This code to visualize motions from saved pkl files
"""

import os
import sys
import numpy as np
import json
import pickle

#by jhugestar
sys.path.append('/ssd/codes/glvis_python/')
from Visualize_human_gl import showSkeleton #opengl visualization.  showSkeleton(skelNum, dim, frames)

path='/ssd/data/panoptic-toolbox/data_haggling'

speaking_path='./speaking_annotation'

# """Visualze pkl raw file (all people of a sequence in pkl file)"""
# outputFolder='./panopticDB_pkl'

# seqFiles=[ os.path.join(outputFolder,f) for f in sorted(list(os.listdir(outputFolder))) ]
# for seqPath in seqFiles:
#     #The motion data of all people in this sequence is saved here
#     print(seqPath)
#     motionData = pickle.load( open( seqPath, "rb" ) )
#     #motionData[0]['joints19'].shape  #(dim,frames)
#     showSkeleton([motionData[0]['joints19'], motionData[1]['joints19'], motionData[2]['joints19']])


"""Visualze a haggling game file (three people of the game in pkl file)"""
outputFolder='./panopticDB_pkl_hagglingProcessed'

seqFiles=[ os.path.join(outputFolder,f) for f in sorted(list(os.listdir(outputFolder))) ]
for seqPath in seqFiles:
    #The motion data of all people in this sequence is saved here
    print(seqPath)
    motionData = pickle.load( open( seqPath, "rb" ) )
    #motionData[0]['joints19'].shape  #(dim,frames)
    
    ##Audio data
    #speak_annot = speaking_path + '/170221_haggling_b1_scene0.txt'
    # f = open(speak_annot)
    # words = f.read().split() #ID, Word, Start, -, End
    # num = len(words) /5
    # if(num !=len(words) /5):
    #     print('Unexpected token num. len(words)=={0}'.format(len(words)))
    #     break

    # for i in range(num):
    #     if words[i*5+0] =="default":
    #         id = -1
    #     else:
    #         id = int(words[i*5+0])
    #     word = words[i*5+1]
    #     startTime = words[i*5+2]
    #     startTime_sec = float(startTime[:2])* 3600 + float(startTime[3:5])* 60 + float(startTime[6:])
    #     endTime = words[i*5+4]
    #     endTime_sec = float(endTime[:2])* 3600 + float(endTime[3:5])* 60 + float(endTime[6:])
    #     print("{0}=={1} -> {2}=={3}".format(startTime,startTime_sec,endTime,endTime_sec))

    

    showSkeleton([motionData['subjects'][0]['joints19'], motionData['subjects'][1]['joints19'], motionData['subjects'][2]['joints19']])