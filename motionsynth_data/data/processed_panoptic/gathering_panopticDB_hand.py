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
outputFolder='./panopticDB_hand_pkl'

if not os.path.exists(outputFolder):
    os.mkdir(outputFolder)

seqPathSet=[ os.path.join(path,f) for f in sorted(list(os.listdir(path))) ]

for seqPath in seqPathSet:

    #The motion data of all people in this sequence is saved here
    motionData_left = list()     #Left hand
    motionData_right = list()   #Right hand

    seqName = os.path.basename(seqPath)

    if os.path.exists("{0}/{1}.pkl".format(outputFolder,seqName)):
         continue

    seqPath = seqPath  + '/hdHand3d'
    seqPathFull=[ os.path.join(seqPath,f) for f in sorted(list(os.listdir(seqPath))) if os.path.isfile(os.path.join(seqPath,f)) and f.endswith('.json') ]

    for i, frameFilePath  in enumerate(seqPathFull):
        
        #Extract frameIdx from fileName
        fileName = os.path.basename(frameFilePath)
        numStartIdx = fileName.find('_') + 3 #'_HD...#
        frameIdx = int(fileName[numStartIdx:numStartIdx+8]) #Always, body3DScene_%08d.json
    
        print(frameFilePath)
        with open(frameFilePath) as cfile:
 
            jsonData = json.load(cfile)

            for target in ('left','right'):
                
                for pose in jsonData['people']:
                    humanId = pose['id']
                    if humanId<0:
                        continue

                    if target=='left':
                        if not ('left_hand' in pose):
                            continue
                        handData = pose['left_hand']
                        motionData = motionData_left #just reference copy. No deep copy is done
                    else: 
                        if not ('right_hand' in pose):
                                continue
                        handData = pose['right_hand']
                        motionData = motionData_right #just reference copy. No deep copy is done

                    validityCheck = np.mean(handData['averageScore'])
                    if validityCheck<0.001:
                        continue

                    #print(humanId)
                    #Add new human dic
                    if humanId >= len(motionData):
                        while humanId >= len(motionData):
                            motionData.append(dict())   #add blank dict
                            motionData[-1]['bValid'] = False
                        
                        #Initialze Currnet Human Data
                        motionData[humanId]['bValid'] = True
                        motionData[humanId]['scores'] = np.empty((21,0),float)
                        motionData[humanId]['reproErrors'] = np.empty((21,0),float)
                        motionData[humanId]['visibilityCnt'] = np.empty((21,0),int)
                        motionData[humanId]['hand21'] = np.empty((63,0),float) #63 = 21x3
                        motionData[humanId]['startFrame'] = frameIdx
                        motionData[humanId]['bValidFrame'] = np.empty((1,0),bool)  #Validity signal

                    elif motionData[humanId]['bValid']==False:       #Already added, but was not valid
                        #Initialze Currnet Human Data
                        motionData[humanId]['bValid'] = True
                        motionData[humanId]['scores'] = np.empty((21,0),float)
                        motionData[humanId]['reproErrors'] = np.empty((21,0),float)
                        motionData[humanId]['visibilityCnt'] = np.empty((21,0),int)
                        motionData[humanId]['hand21'] = np.empty((63,0),float) #63 = 21x3
                        motionData[humanId]['startFrame'] = frameIdx
                        motionData[humanId]['bValidFrame'] = np.empty((1,0),bool)  #Validity signal
                        
                        
                    hand21 = np.array(handData['landmarks']) #(63,)
                    hand21 = hand21.flatten()[:,np.newaxis] #(63,1)

                    scores = np.array(handData['averageScore'])  #(21,)
                    scores = scores[:,np.newaxis] #(21,1)
                    reproError = np.array(handData['averageReproError'])  #(21,)
                    reproError = reproError[:,np.newaxis] #(21,1)
                    visibility = np.array(handData['visibility'])
                    visibilityCnt = np.array([len(x) for x in visibility]) #(21,)
                    visibilityCnt = visibilityCnt[:,np.newaxis] #(21,1)

                    #Append. #This assume that human skeletons exist continuously. Will be broken if drops happen. No this results are happening?
                    localIdx = frameIdx - motionData[humanId]['startFrame'] #zero based inx
                    #assert(motionData[humanId]['joints19'].shape[1] ==localIdx)
                    if motionData[humanId]['hand21'].shape[1] != localIdx:
                        #print('{0} vs {1}'.format(motionData[humanId]['hand21'].shape[1],localIdx))

                        #Add invalid
                        while motionData[humanId]['hand21'].shape[1] !=localIdx:

                            #print('adding: {0} vs {1}'.format(motionData[humanId]['hand21'].shape[1],localIdx))

                            motionData[humanId]['hand21']  = np.append(motionData[humanId]['hand21'], np.zeros((63,1),dtype=float), axis=1) #(63, frames)
                            motionData[humanId]['scores']  = np.append(motionData[humanId]['scores'], np.zeros((21,1),dtype=float), axis=1) #(21, frames)

                            motionData[humanId]['reproErrors']  = np.append(motionData[humanId]['reproErrors'], np.zeros((21,1),dtype=float), axis=1) #(21, frames)
                            motionData[humanId]['visibilityCnt']  = np.append(motionData[humanId]['scores'], np.zeros((21,1),dtype=int), axis=1) #(21, frames)
                            motionData[humanId]['bValidFrame'] = np.append(motionData[humanId]['bValidFrame'], False)  #Validity signal

                    motionData[humanId]['hand21']  = np.append(motionData[humanId]['hand21'],hand21, axis=1) #(63, frames)
                    motionData[humanId]['scores']  = np.append(motionData[humanId]['scores'],scores, axis=1) #(21, frames)
                    motionData[humanId]['reproErrors']  = np.append(motionData[humanId]['reproErrors'],reproError, axis=1) #(21, frames)
                    motionData[humanId]['visibilityCnt']  = np.append(motionData[humanId]['scores'],scores, axis=1) #(21, frames)
                    motionData[humanId]['bValidFrame'] = np.append(motionData[humanId]['bValidFrame'], True)  #Validity signal

                    # if motionData[humanId]['face70'].shape[1] == 1045:
                    #     print('{0} vs {1}'.format(motionData[humanId]['face70'].shape[1],localIdx+1))
                    

    #print(motionData)
    pickle.dump( {'left': motionData_left, 'right':motionData_right}, open( "{0}/{1}.pkl".format(outputFolder,seqName), "wb" ) )