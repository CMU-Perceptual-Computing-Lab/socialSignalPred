import os
import sys
import numpy as np
import scipy.io as io
import random
import os
import cPickle as pickle
import matplotlib.pyplot as plt

#by jhugestar
sys.path.append('/ssd/codes/glvis_python/')
#from glViewer import SetFaceParmData,setSpeech,setSpeechGT,setSpeech_binary, setSpeechGT_binary, init_gl #opengl visualization 
import glViewer



fileName = '170407_haggling_b2_group0.pkl'  #interesting


# The following is for drawbody_joint73 (Holden's format)
sys.path.append('/ssd/codes/glvis_python/')
from Quaternions import Quaternions
def get_Holden_Data_73(skel_list, ignore_root=False):

    skel_list_output = []
    footsteps_output = []

    for ai in range(len(skel_list)):
        anim = np.swapaxes(skel_list[ai].copy(), 0, 1)  # frameNum x 73
        
        joints, root_x, root_z, root_r = anim[:,:-7], anim[:,-7], anim[:,-6], anim[:,-5]
        joints = joints.reshape((len(joints), -1, 3)) #(frameNum,66) -> (frameNum, 22, 3)
        
        rotation = Quaternions.id(1)
        offsets = []
        translation = np.array([[0,0,0]])
        
        if not ignore_root:
            for i in range(len(joints)):
                joints[i,:,:] = rotation * joints[i]
                joints[i,:,0] = joints[i,:,0] + translation[0,0]
                joints[i,:,2] = joints[i,:,2] + translation[0,2]
                rotation = Quaternions.from_angle_axis(-root_r[i], np.array([0,1,0])) * rotation
                offsets.append(rotation * np.array([0,0,1]))
                translation = translation + rotation * np.array([root_x[i], 0, root_z[i]])
        
        #joints dim:(frameNum, 22, 3)
        #Scaling        
        joints[:,:,:] = joints[:,:,:] *5 #m -> cm
        joints[:,:,1] = joints[:,:,1] *-1 #Flip Y axis
        
        #Reshaping        
        joints = joints.reshape(joints.shape[0], joints.shape[1]*joints.shape[2]) # frameNum x 66
        joints =  np.swapaxes(joints, 0, 1)  # 66  x frameNum

        skel_list_output.append(joints)
        footsteps_output.append(anim[:,-4:])
    
    skel_list_output = np.asarray(skel_list_output)

    return skel_list_output


#read speech
seqPath = '/ssd/codes/haggling_audio/panopticDB_pkl_speech_hagglingProcessed/' +fileName
motionData = pickle.load( open( seqPath, "rb" ) )
speechData = [motionData['speechData'][0], motionData['speechData'][1], motionData['speechData'][2]]



#Read Face mesh
##read face mesh parameter
seqPath = '/ssd/codes/pytorch_motionSynth/motionsynth_data/data/processed_panoptic/panopticDB_faceMesh_pkl_hagglingProcessed/' +fileName
faceData = pickle.load( open( seqPath, "rb" ) )
faceData = faceData['subjects'] 


# #Draw Speech vs Face
# fig = plt.figure()
# for pIdx  in range(len(faceData)):

#     ax = plt.subplot(3,1,pIdx+1)


#     #Draw speech component
#     ax.plot(speechData[pIdx]['indicator'][:],label='speech')
    

#     #Draw face component
#     for comp  in range(7):
#         ax.plot(faceData[pIdx]['face_exp'][comp,:],label='faceComp: {}'.format(comp))
#         ax.hold(True)

#     ax.legend()
#     #ax.grid()
# #plt.show()



#Read body, holden's format
fileName_npz = fileName.replace('pkl','npz')
X = np.load('/ssd/codes/pytorch_motionSynth/motionsynth_data/data/processed/panoptic_npz/' + fileName_npz)['clips'] #(17944, 240, 73)
X = np.swapaxes(X, 1, 2).astype(np.float32) #(17944, 73, 240)
bodyData = get_Holden_Data_73([ X[0,:,:], X[1,:,:], X[2,:,:] ], ignore_root=True) # bodyData: 3 x (66,frames  )


fig = plt.figure()
for pIdx  in range(len(faceData)):

    ax = plt.subplot(3,1,pIdx+1)


    #Draw speech component
    ax.plot(speechData[pIdx]['indicator'][:],label='speech')
    

    #Draw face component
    for comp  in [13, 16, 20]: #left and right hand

        # for i, axis in enumerate(['x','y','z']):
        #     data = np.swapaxes(bodyData[pIdx][(comp*3+i):(comp*3+i+1),:],0,1)
            
        #     ax.plot(data,label='joint{}_{}'.format(comp, axis))

        #     print('joint{}_{}'.format(comp, axis))

        #ax.hold(True)
        #Compute velocity
        data_vel =bodyData[pIdx][(comp*3):(comp*3+3),:]
        data_vel = data_vel[:,1:]  -  data_vel[:,:-1]
        data_vel = sum(data_vel**2)**0.5
        ax.plot(data_vel,label='joint{}_vel'.format(comp))


    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #ax.grid()
plt.show()
