""" This function is to load a sequence data (facewWarehouse Parameters) in panoptic db (txt files for frames),
 and combining the motions from the same person as a single matrix
 and saving all of them as a single file.
"""

import os
import numpy as np
import json
import cPickle as pickle
            
#path='/ssd/data/panoptic-toolbox/data_haggling'
path='/posefs0c/panoptic/'
#seqName='170221_haggling_b1'
outputFolder='./panopticDB_faceMesh_pkl'

if not os.path.exists(outputFolder):
    os.mkdir(outputFolder)

seqPathSet=[ os.path.join(path,f) for f in sorted(list(os.listdir(path))) ]

def ConvLine2Float(data):
    data = data.split(' ')[:-1] #Ignore the last element (\r\n)
    data = [ float(x) for x in data]

    return np.array(data)


def ReadFaceParams(faceParamFile):
    #faceParamFile = '/ssd/data/panoptic-toolbox/data_haggling/170224_haggling_a1/meshTrack_face/meshTrack_00000933.txt'

    fp = open(faceParamFile, "r")
    line = fp.readline() #'ver 1.30\r\n'
    peopleNum = int(fp.readline()) #num of people

    faceParam_all =[]

    for num in range(peopleNum):
        humanIdx = fp.readline() #humanIdx
        humanIdx = int(humanIdx)
        buffer = fp.readline() #ADAM_Body 0
        buffer = fp.readline() #SMPL_Body 0
        buffer = fp.readline() #LHand 0
        buffer = fp.readline() #RHand 0
        buffer = fp.readline() #Face 3 1 3 1 3 1 150 1 200 1

        trans = ConvLine2Float(fp.readline()) #trans 3x1
        rot = ConvLine2Float(fp.readline()) #rot 3x1
        pivot = ConvLine2Float(fp.readline()) #rot_pivot 3x1

        face_id = ConvLine2Float(fp.readline()) #face_id 150x1
        # face_id = face_id.split(' ')[:150]
        # face_id = [ float(x) for x in face_id]
        #face_id = np.array(face_id)

        face_exp = ConvLine2Float(fp.readline()) #face_express 200x1
        # face_exp = face_exp.split(' ')[:200]
        # face_exp = [ float(x) for x in face_exp]
        #face_exp = np.array(face_exp)

        buffer = fp.readline() #garbage
        buffer = fp.readline() #blank

        faceParam_all.append( {'humanId':humanIdx, 'trans':trans, 'rot':rot, 'rot_pivot':pivot, 'face_id':face_id, 'face_exp': face_exp})

    fp.close()
    
    return faceParam_all


for seqPath in seqPathSet:

    #The motion data of all people in this sequence is saved here
    motionData = list()     
    
    seqName = os.path.basename(seqPath)

    if os.path.exists("{0}/{1}.pkl".format(outputFolder,seqName)):
         continue

    seqPath = seqPath  + '/meshTrack_face'

    if not os.path.exists(seqPath):
        continue

    seqPathFull=[ os.path.join(seqPath,f) for f in sorted(list(os.listdir(seqPath))) if os.path.isfile(os.path.join(seqPath,f)) and f.endswith('.txt') ]

    for i, frameFilePath  in enumerate(seqPathFull):
        
        #Extract frameIdx from fileName
        print(frameFilePath)
        fileName = os.path.basename(frameFilePath)
        numStartIdx = fileName.find('_') + 1 #'_HD...#
        frameIdx = int(fileName[numStartIdx:numStartIdx+8]) #Always, body3DScene_%08d.json
               

        faceParam_all = ReadFaceParams(frameFilePath)

        dupCheck=[]
        for faceParm in faceParam_all:
            #print(faceParm['humanId'])
            humanId = faceParm['humanId']

            if humanId in dupCheck:
                continue    #already added...
            else:
                dupCheck.append(humanId)

            #Add new human dic
            if humanId >= len(motionData):
                while humanId >= len(motionData):
                    motionData.append(dict())   #add blank dict
                    motionData[-1]['bValid'] = False
                
                #Initialze Currnet Human Data
                motionData[humanId]['bValid'] = True
                motionData[humanId]['trans'] = np.empty((3,0),float)
                motionData[humanId]['rot'] = np.empty((3,0),float)
                motionData[humanId]['rot_pivot'] = np.empty((3,0),int)
                motionData[humanId]['face_id'] = np.empty((150,0),float) 
                motionData[humanId]['face_exp'] = np.empty((200,0),float) 

                motionData[humanId]['startFrame'] = frameIdx
                motionData[humanId]['bValidFrame'] = np.empty((1,0),bool)  #Validity signal

            elif motionData[humanId]['bValid']==False:       #Already added, but was not valid
                 #Initialze Currnet Human Data
                motionData[humanId]['bValid'] = True
                motionData[humanId]['trans'] = np.empty((3,0),float)
                motionData[humanId]['rot'] = np.empty((3,0),float)
                motionData[humanId]['rot_pivot'] = np.empty((3,0),int)
                motionData[humanId]['face_id'] = np.empty((150,0),float) 
                motionData[humanId]['face_exp'] = np.empty((200,0),float) 

                motionData[humanId]['startFrame'] = frameIdx
                motionData[humanId]['bValidFrame'] = np.empty((1,0),bool)  #Validity signal
                
                
            face_id = np.array(faceParm['face_id']) #(150,)
            face_id = face_id.flatten()[:,np.newaxis] #(150,1)

            face_exp = np.array(faceParm['face_exp']) #(200,)
            face_exp = face_exp.flatten()[:,np.newaxis] #(200,1)

            trans = np.array(faceParm['trans']) #(3,)
            trans = trans[:,np.newaxis] #(3,1)

            rot = np.array(faceParm['rot']) #(3,)
            rot = rot[:,np.newaxis] #(3,1)

            rot_pivot = np.array(faceParm['rot_pivot']) #(3,)
            rot_pivot = rot_pivot[:,np.newaxis] #(3,1)

            #Append. 
            localIdx = frameIdx - motionData[humanId]['startFrame'] #zero based inx
            #assert(motionData[humanId]['joints19'].shape[1] ==localIdx)
            if motionData[humanId]['trans'].shape[1] != localIdx:
                #print('{0} vs {1}'.format(motionData[humanId]['face70'].shape[1],localIdx))

                #Add invalid
                while motionData[humanId]['trans'].shape[1] !=localIdx:

                    #print('adding: {0} vs {1}'.format(motionData[humanId]['face70'].shape[1],localIdx))

                    motionData[humanId]['face_id']  = np.append(motionData[humanId]['face_id'], np.zeros((150,1),dtype=float), axis=1) #(150, frames)
                    motionData[humanId]['face_exp']  = np.append(motionData[humanId]['face_exp'], np.zeros((200,1),dtype=float), axis=1) #(200, frames)

                    motionData[humanId]['trans']  = np.append(motionData[humanId]['trans'], np.zeros((3,1),dtype=float), axis=1) #(150, frames)
                    motionData[humanId]['rot']  = np.append(motionData[humanId]['rot'], np.zeros((3,1),dtype=float), axis=1) #(150, frames)
                    motionData[humanId]['rot_pivot']  = np.append(motionData[humanId]['rot_pivot'], np.zeros((3,1),dtype=float), axis=1) #(150, frames)


            motionData[humanId]['face_id']  = np.append(motionData[humanId]['face_id'],face_id, axis=1) #(150, frames)
            motionData[humanId]['face_exp']  = np.append(motionData[humanId]['face_exp'],face_exp, axis=1) #(200, frames)

            motionData[humanId]['trans']  = np.append(motionData[humanId]['trans'],trans, axis=1) #(3, frames)
            motionData[humanId]['rot']  = np.append(motionData[humanId]['rot'],rot, axis=1) #(3, frames)
            motionData[humanId]['rot_pivot']  = np.append(motionData[humanId]['rot_pivot'],rot_pivot, axis=1) #(3, frames)
            motionData[humanId]['bValidFrame'] = np.append(motionData[humanId]['bValidFrame'], True)  #Validity signal

            # if motionData[humanId]['face70'].shape[1] == 1045:
            #     print('{0} vs {1}'.format(motionData[humanId]['face70'].shape[1],localIdx+1))
                    

    #print(motionData)
    pickle.dump( motionData, open( "{0}/{1}.pkl".format(outputFolder,seqName), "wb" ) )