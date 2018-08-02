""" This function is to load a sequence data in panoptic db (json files for frames),
 and combining the motions from the same person as a single matrix
 and saving all of them as a single file.
 Combining motions from the same 
"""

import os
import numpy as np
import json
import pickle
            
path='/ssd/data/panoptic-toolbox/data_haggling'
#seqName='170221_haggling_b1'
outputFolder='./panopticDB'

if not os.path.exists(outputFolder):
    os.mkdir(outputFolder)

seqPathSet=[ os.path.join(path,f) for f in sorted(list(os.listdir(path))) ]

for seqPath in seqPathSet:

    #The motion data of all people in this sequence is saved here
    motionData = list()

    seqName = os.path.basename(seqPath)
    seqPath = seqPath  + '/hdPose3d_stage1_coco19'
    seqPathFull=[ os.path.join(seqPath,f) for f in sorted(list(os.listdir(seqPath))) ]

    for i, frameFilePath  in enumerate(seqPathFull):
        
        #Extract frameIdx from fileName
        fileName = os.path.basename(frameFilePath)
        numStartIdx = fileName.find('_') + 1
        frameIdx = int(fileName[numStartIdx:numStartIdx+8]) #Always, body3DScene_%08d.json
    
        print(frameFilePath)
        with open(frameFilePath) as cfile:
 
            jsonData = json.load(cfile)

            for pose in jsonData['bodies']:
                humanId = pose['id']
                #print(humanId)
                #Add new human dic
                if humanId >= len(motionData):
                    while humanId >= len(motionData):
                        motionData.append(dict())   #add blank dict
                        motionData[-1]['bValid'] = False
                    
                    #Initialze Currnet Human Data
                    motionData[humanId]['bValid'] = True
                    motionData[humanId]['scores'] = np.empty((19,0),float)
                    motionData[humanId]['joints19'] = np.empty((57,0),float) #57 = 19x3
                    motionData[humanId]['startFrame'] = frameIdx
                    
                joints19 = np.array(pose['joints19']) #(76,)
                joints19 = joints19.reshape(-1,4) #19x4 where last column has recon. scores
                scores = joints19[:,3:4] #19x1
                joints19 = joints19[:,:3] #19x3
                joints19 = joints19.flatten()[:,np.newaxis] #(57,1)

                #Append. #This assume that human skeletons exist continuously. Will be broken if drops happen. No this results are happening?
                localIdx = frameIdx - motionData[humanId]['startFrame'] #zero based inx
                #assert(motionData[humanId]['joints19'].shape[1] ==localIdx)
                if motionData[humanId]['joints19'].shape[1] != localIdx:
                    print('{0} vs {1}'.format(motionData[humanId]['joints19'].shape[1],localIdx))
                    assert(False)

                motionData[humanId]['joints19']  = np.append(motionData[humanId]['joints19'],joints19, axis=1) #(57, frames)
                motionData[humanId]['scores']  = np.append(motionData[humanId]['scores'],scores, axis=1) #(19, frames)
    #print(motionData)
    pickle.dump( motionData, open( "{0}/{1}.pkl".format(outputFolder,seqName), "wb" ) )
