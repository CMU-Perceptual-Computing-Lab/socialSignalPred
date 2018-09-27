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
outputFolder='./panopticDB_face_pkl'

if not os.path.exists(outputFolder):
    os.mkdir(outputFolder)

seqPathSet=[ os.path.join(path,f) for f in sorted(list(os.listdir(path))) ]

for seqPath in seqPathSet:

    #The motion data of all people in this sequence is saved here
    motionData = list()

    seqName = os.path.basename(seqPath)

    if os.path.exists("{0}/{1}.pkl".format(outputFolder,seqName)):
         continue


    seqPath = seqPath  + '/hdFace3d'
    seqPathFull=[ os.path.join(seqPath,f) for f in sorted(list(os.listdir(seqPath))) if os.path.isfile(os.path.join(seqPath,f))]

    for i, frameFilePath  in enumerate(seqPathFull):
        
        #Extract frameIdx from fileName
        fileName = os.path.basename(frameFilePath)
        numStartIdx = fileName.find('_') + 3 #'_HD...#
        frameIdx = int(fileName[numStartIdx:numStartIdx+8]) #Always, body3DScene_%08d.json
    
        print(frameFilePath)
        with open(frameFilePath) as cfile:
 
            jsonData = json.load(cfile)

            for pose in jsonData['people']:
                humanId = pose['id']
                if humanId<0:
                    continue

                faceData = pose['face70']

                validityCheck = np.mean(faceData['averageScore'])
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
                    motionData[humanId]['scores'] = np.empty((70,0),float)
                    motionData[humanId]['reproErrors'] = np.empty((70,0),float)
                    motionData[humanId]['visibilityCnt'] = np.empty((70,0),int)
                    motionData[humanId]['face70'] = np.empty((210,0),float) #210 = 70x3
                    motionData[humanId]['startFrame'] = frameIdx
                    motionData[humanId]['bValidFrame'] = np.empty((1,0),bool)  #Validity signal

                elif motionData[humanId]['bValid']==False:       #Already added, but was not valid
                    #Initialze Currnet Human Data
                    motionData[humanId]['bValid'] = True
                    motionData[humanId]['scores'] = np.empty((70,0),float)
                    motionData[humanId]['reproErrors'] = np.empty((70,0),float)
                    motionData[humanId]['visibilityCnt'] = np.empty((70,0),int)
                    motionData[humanId]['face70'] = np.empty((210,0),float) #210 = 70x3
                    motionData[humanId]['startFrame'] = frameIdx
                    motionData[humanId]['bValidFrame'] = np.empty((1,0),bool)  #Validity signal
                    
                    
                face70 = np.array(faceData['landmarks']) #(210,)
                face70 = face70.flatten()[:,np.newaxis] #(210,1)

                scores = np.array(faceData['averageScore'])  #(70,)
                scores = scores[:,np.newaxis] #(70,1)
                reproError = np.array(faceData['averageReproError'])  #(70,)
                reproError = reproError[:,np.newaxis] #(70,1)
                visibility = np.array(faceData['visibility'])
                visibilityCnt = np.array([len(x) for x in visibility]) #(70,)
                visibilityCnt = visibilityCnt[:,np.newaxis] #(70,1)

                #Append. #This assume that human skeletons exist continuously. Will be broken if drops happen. No this results are happening?
                localIdx = frameIdx - motionData[humanId]['startFrame'] #zero based inx
                #assert(motionData[humanId]['joints19'].shape[1] ==localIdx)
                if motionData[humanId]['face70'].shape[1] != localIdx:
                    #print('{0} vs {1}'.format(motionData[humanId]['face70'].shape[1],localIdx))

                    #Add invalid
                    while motionData[humanId]['face70'].shape[1] !=localIdx:

                        #print('adding: {0} vs {1}'.format(motionData[humanId]['face70'].shape[1],localIdx))

                        motionData[humanId]['face70']  = np.append(motionData[humanId]['face70'], np.zeros((210,1),dtype=float), axis=1) #(210, frames)
                        motionData[humanId]['scores']  = np.append(motionData[humanId]['scores'], np.zeros((70,1),dtype=float), axis=1) #(70, frames)

                        motionData[humanId]['reproErrors']  = np.append(motionData[humanId]['reproErrors'], np.zeros((70,1),dtype=float), axis=1) #(70, frames)
                        motionData[humanId]['visibilityCnt']  = np.append(motionData[humanId]['scores'], np.zeros((70,1),dtype=int), axis=1) #(70, frames)
                        motionData[humanId]['bValidFrame'] = np.append(motionData[humanId]['bValidFrame'], False)  #Validity signal

                motionData[humanId]['face70']  = np.append(motionData[humanId]['face70'],face70, axis=1) #(210, frames)
                motionData[humanId]['scores']  = np.append(motionData[humanId]['scores'],scores, axis=1) #(70, frames)
                motionData[humanId]['reproErrors']  = np.append(motionData[humanId]['reproErrors'],reproError, axis=1) #(70, frames)
                motionData[humanId]['visibilityCnt']  = np.append(motionData[humanId]['scores'],scores, axis=1) #(70, frames)
                motionData[humanId]['bValidFrame'] = np.append(motionData[humanId]['bValidFrame'], True)  #Validity signal

                # if motionData[humanId]['face70'].shape[1] == 1045:
                #     print('{0} vs {1}'.format(motionData[humanId]['face70'].shape[1],localIdx+1))
                

    #print(motionData)
    pickle.dump( motionData, open( "{0}/{1}.pkl".format(outputFolder,seqName), "wb" ) )
