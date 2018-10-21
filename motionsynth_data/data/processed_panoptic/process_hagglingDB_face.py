""" 
    This code does processing for haggling sequences
    1: Align the startFarmes for the people in the same group
    2: Separate the name of pkl file, to have a single group only
"""

import os
import sys
import numpy as np
import json
import pickle
import copy

#by jhugestar
sys.path.append('/ssd/codes/glvis_python/')
from glViewer import showSkeleton #opengl visualization.  showSkeleton(skelNum, dim, frames)

path='/ssd/data/panoptic-toolbox/data_haggling'
inputFolder='./panopticDB_face_pkl'
outputFolder='./panopticDB_face_pkl_hagglingProcessed'

if not os.path.exists(outputFolder):
    os.mkdir(outputFolder)

seqFiles=[ os.path.join(inputFolder,f) for f in sorted(list(os.listdir(inputFolder))) ]




'''
    Input:
        face_list:  faceNum x element['face70'](210, frames)
    Output:
        faceNormal_list:
        face_list:  faceNum x (3, frames)
'''
def ComputeFaceNormal(faceData):

    leftEye = faceData[(45*3):(45*3+3),:].transpose() #210xframes
    rightEye = faceData[(36*3):(36*3+3),:].transpose() #210xframes
    nose = faceData[(33*3):(33*3+3),:].transpose() #210xframes
    
    left2Right = rightEye - leftEye
    right2nose = nose - rightEye

    from sklearn.preprocessing import normalize
    left2Right = normalize(left2Right, axis=1)
    #Check: np.linalg.norm(left2Right,axis=1)
    right2nose = normalize(right2nose, axis=1)

    faceNormal = np.cross(left2Right,right2nose)
    faceNormal[:,1] = 0 #Project on x-z plane, ignoring y axis
    faceNormal = normalize(faceNormal, axis=1)

    return faceNormal



#Load Haggling Game info
hagglingInfoFilePath = './annot_domedb_180223_30db_processed_win.json'
with open(hagglingInfoFilePath) as cfile:
    hagglingInfo = json.load(cfile)
    hagglingInfo = hagglingInfo['data']

for seqInfo in hagglingInfo:
    seqName = seqInfo['seqName']
    seqPath = os.path.join(inputFolder,seqName+'.pkl')

    if not os.path.exists(seqPath):
        print('Does not exists: {0}'.format(seqPath))
        continue

    #The motion data of all people in this sequence is saved here
    print(seqPath)
    motionData = pickle.load( open( seqPath, "rb" ) )

    """Debug"""
    #motionData[0]['joints19'].shape  #(dim,frames)
    #showSkeleton([motionData[0]['joints19'], motionData[1]['joints19'], motionData[2]['joints19']])

    
    for groupNum, groupInfo in enumerate(seqInfo['scenes']):

        if os.path.exists("{0}/{1}_group{2}.pkl".format(outputFolder,seqName,groupNum)):
            continue

        #print(groupInfo.keys())
        groupStartFrame = groupInfo['rangeStart_hd']
        groupEndFrame = groupInfo['rangeEnd_hd']
        buyerId = groupInfo['team'][0]
        sellerIds = groupInfo['team'][1]
        winnerId = groupInfo['winnerIds']
        loserId = copy.deepcopy(sellerIds)
        loserId.remove(winnerId)
        loserId = loserId[0]
        leftSellerId = groupInfo['sellerLR'][0]
        rightSellerId = groupInfo['sellerLR'][1]

        #Find a group of people
        group = list()
        #for humanId in (buyerId,leftSellerId,rightSellerId): #buyer, leftSeller, rightSeller order
        for humanId in (buyerId,winnerId,loserId): #buyer, leftSeller, rightSeller order

            if humanId >=len(motionData):
                group = list()

                print('{0}_group{1}: humanId{2} >=len(motionData) {3}'.format(seqName,groupNum,humanId,len(motionData)))
                break
            group.append(motionData[humanId])
            localStartFrame = groupStartFrame  - group[-1]['startFrame']
            localEndFrame = groupEndFrame - group[-1]['startFrame']
            group[-1]['face70'] = group[-1]['face70'][:, localStartFrame:localEndFrame]
            group[-1]['scores'] = group[-1]['scores'][:, localStartFrame:localEndFrame]            
            group[-1]['humanId'] = humanId

            #Compute Face Normal
            faceNormal = ComputeFaceNormal(group[-1]['face70']) #(1744,3)
            group[-1]['normal'] = np.swapaxes(faceNormal,0,1) #save (1744,3)

        haggling=dict()
        haggling['startFrame'] = groupStartFrame
        haggling['buyerId'] = buyerId
        haggling['sellerIds'] = sellerIds
        haggling['winnerId'] = winnerId
        haggling['leftSellerId'] = leftSellerId
        haggling['rightSellerId'] = rightSellerId
        haggling['subjects'] = group

        #Save the output
        pickle.dump( haggling, open( "{0}/{1}_group{2}.pkl".format(outputFolder,seqName,groupNum), "wb" ) )

        #showSkeleton([haggling['subjects'][0]['joints19'], haggling['subjects'][1]['joints19'], haggling['subjects'][2]['joints19']])



