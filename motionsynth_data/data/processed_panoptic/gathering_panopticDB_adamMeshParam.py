""" This function is to load a sequence data (facewWarehouse Parameters) in panoptic db (txt files for frames),
 and combining the motions from the same person as a single matrix
 and saving all of them as a single file.
"""

import os
import numpy as np
import json
import cPickle as pickle
            
#path='/ssd/data/panoptic-toolbox/data_haggling'
#path='/posefs0c/panoptic/'
#path='/media/posefs3b/Users/xiu/domedb/(seqName)/hdPose3d_Adam_stage0',
path='/media/posefs3b/Users/xiu/domedb/'
#seqName='170221_haggling_b1'
outputFolder='./panopticDB_adamMesh_pkl'

if not os.path.exists(outputFolder):
    os.mkdir(outputFolder)

seqPathSet=[ os.path.join(path,f) for f in sorted(list(os.listdir(path))) if "haggling" in f ]

def ConvLine2Float(data):
    data = data.split(' ')[:-1] #Ignore the last element (\r\n)
    data = [ float(x) for x in data]

    return np.array(data)


for seqPath in seqPathSet:

    #The motion data of all people in this sequence is saved here
    motionData = list()     
    
    seqName = os.path.basename(seqPath)

    if os.path.exists("{0}/{1}.pkl".format(outputFolder,seqName)):
         continue

    seqPath = seqPath  + '/hdPose3d_Adam_stage0'

    if not os.path.exists(seqPath):
        continue

    seqPathFull=[ os.path.join(seqPath,f) for f in sorted(list(os.listdir(seqPath))) if os.path.isfile(os.path.join(seqPath,f)) and f.endswith('.pkl') ]

    for i, frameFilePath  in enumerate(seqPathFull):
        
        #Extract frameIdx from fileName
        print(frameFilePath)
        fileName = os.path.basename(frameFilePath)
        numStartIdx = fileName.find('_') + 1 #'_HD...#
        frameIdx = int(fileName[numStartIdx:numStartIdx+8]) #Always, body3DScene_%08d.json
    
     
        adamParam_all = pickle.load( open( frameFilePath, "rb" ) )


        for adamParam in adamParam_all:
            #print(adamParam['humanId'])
            humanId = adamParam['id']
            #Add new human dic
            if humanId >= len(motionData):
                while humanId >= len(motionData):
                    motionData.append(dict())   #add blank dict
                    motionData[-1]['bValid'] = False
                
                #Initialze Currnet Human Data
                motionData[humanId]['bValid'] = True
                motionData[humanId]['trans'] = np.empty((3,0),float)
                motionData[humanId]['pose'] = np.empty((186,0),float)
                motionData[humanId]['betas'] = np.empty((30,0),int)
                motionData[humanId]['faces'] = np.empty((200,0),float) 

                motionData[humanId]['startFrame'] = frameIdx
                motionData[humanId]['bValidFrame'] = np.empty((1,0),bool)  #Validity signal

            elif motionData[humanId]['bValid']==False:       #Already added, but was not valid
                 #Initialze Currnet Human Data
                motionData[humanId]['bValid'] = True
                motionData[humanId]['trans'] = np.empty((3,0),float)
                motionData[humanId]['pose'] = np.empty((186,0),float)
                motionData[humanId]['betas'] = np.empty((30,0),int)
                motionData[humanId]['faces'] = np.empty((200,0),float) 

                motionData[humanId]['startFrame'] = frameIdx
                motionData[humanId]['bValidFrame'] = np.empty((1,0),bool)  #Validity signal
                
                
            pose = np.array(adamParam['pose']) #(186,)
            pose = pose.flatten()[:,np.newaxis] #(186,1)

            betas = np.array(adamParam['betas']) #(30,)
            betas = betas.flatten()[:,np.newaxis] #(30,1)

            faces = np.array(adamParam['faces']) #(200,)
            faces = faces.flatten()[:,np.newaxis] #(200,1)

            trans = np.array(adamParam['trans']) #(3,)
            trans = trans[:,np.newaxis] #(3,1)

            #Append. 
            localIdx = frameIdx - motionData[humanId]['startFrame'] #zero based inx
            #assert(motionData[humanId]['joints19'].shape[1] ==localIdx)
            if motionData[humanId]['trans'].shape[1] != localIdx:
                #print('{0} vs {1}'.format(motionData[humanId]['face70'].shape[1],localIdx))

                #Add invalid
                while motionData[humanId]['trans'].shape[1] !=localIdx:

                    #print('adding: {0} vs {1}'.format(motionData[humanId]['face70'].shape[1],localIdx))

                    motionData[humanId]['faces']  = np.append(motionData[humanId]['faces'], np.zeros((200,1),dtype=float), axis=1) #(200, frames)
                    motionData[humanId]['betas']  = np.append(motionData[humanId]['betas'], np.zeros((30,1),dtype=float), axis=1) #(30, frames)
                    motionData[humanId]['pose']  = np.append(motionData[humanId]['pose'], np.zeros((186,1),dtype=float), axis=1) #(186, frames)

                    motionData[humanId]['trans']  = np.append(motionData[humanId]['trans'], np.zeros((3,1),dtype=float), axis=1) #(3, frames)


            motionData[humanId]['trans']  = np.append(motionData[humanId]['trans'],trans, axis=1) #(3, frames)

            motionData[humanId]['faces']  = np.append(motionData[humanId]['faces'],faces, axis=1) #(200, frames)
            motionData[humanId]['betas']  = np.append(motionData[humanId]['betas'],betas, axis=1) #(200, frames)
            motionData[humanId]['pose']  = np.append(motionData[humanId]['pose'],pose, axis=1) #(200, frames)

            motionData[humanId]['bValidFrame'] = np.append(motionData[humanId]['bValidFrame'], True)  #Validity signal

            # if motionData[humanId]['face70'].shape[1] == 1045:
            #     print('{0} vs {1}'.format(motionData[humanId]['face70'].shape[1],localIdx+1))
                    

    #print(motionData)
    pickle.dump( motionData, open( "{0}/{1}.pkl".format(outputFolder,seqName), "wb" ) )