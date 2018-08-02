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
    showSkeleton([motionData['subjects'][0]['joints19'], motionData['subjects'][1]['joints19'], motionData['subjects'][2]['joints19']])