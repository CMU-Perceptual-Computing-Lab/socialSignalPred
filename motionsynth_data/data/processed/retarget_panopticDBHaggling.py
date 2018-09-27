'''
This code is to convert PanopticDB-Haggling (pkl files for each group) to separate CMU mocap files
'''
import re
import sys
import numpy as np
import scipy.io as io

sys.path.append('../../motion')
import BVH as BVH

import pickle

import Animation as Animation
from Quaternions import Quaternions
# from Pivots import Pivots
from InverseKinematics import BasicJacobianIK, JacobianInverseKinematics

import os


#by jhugestar
sys.path.append('/ssd/codes/glvis_python/')
from Visualize_human_gl import showSkeleton #opengl visualization 

def conv_visGl_form(rest_targets): #skel: (frames, 31, 3)

    rest_targets = rest_targets.reshape(rest_targets.shape[0],rest_targets.shape[1]*rest_targets.shape[2]) #(frames, 93)
    rest_targets = np.swapaxes(rest_targets,0,1)  #(93,frames)

    return rest_targets


database = []

sourcePath='/ssd/codes/pytorch_motionSynth/motionsynth_data/data/processed_panoptic/panopticDB_pkl_hagglingProcessed'

pkl_files=[ os.path.join(sourcePath,f) for f in sorted(list(os.listdir(sourcePath))) ]

outputFolder = './panoptic_speak/'
if not os.path.exists(outputFolder):
    os.mkdir(outputFolder)


for i, filePath in enumerate(pkl_files):
    
    seqName = os.path.splitext(os.path.basename(filePath))[0]
    print('%i of %i Processing %s' % (i+1, len(pkl_files), filePath))

    motionData = pickle.load( open( filePath, "rb" ) )
    
    #cdf = pycdf.CDF('/media/hanbyulj/ssd/data/h36m/S7/MyPoseFeatures/D3_Positions_mono_universal/Directions.54138969.cdf')

    for pid, subjectInfo in enumerate(motionData['subjects']): #pid = 0,1, or 2. (Not humanId)

        targetFile = outputFolder + seqName + '_p' + str(pid)+ '.bvh'
        if os.path.exists(targetFile):
            continue


        normalizedPose = subjectInfo['joints19']  #(57,frames)
        normalizedPose = np.transpose(normalizedPose) #(frames,57)
        
        normalizedPose = normalizedPose.reshape(normalizedPose.shape[0],19,3) #(frames,19,3)

        # # """DEBUG"""
        # normalizedPose = normalizedPose[:2000,:,:]
        
        #Flip Y axis
        normalizedPose[:,:,1] = -normalizedPose[:,:,1]

        #normalizedPose = normalizedPose * 0.2   #rescaling
        #panopticHeight = max(normalizedPose[:,:,1].flatten()) - min(normalizedPose[:,:,1].flatten()) 
        panopticThigh = normalizedPose[:,6,:] - normalizedPose[:,7,:]
        panopticThigh = panopticThigh**2
        panopticHeight = np.mean(np.sqrt(np.sum(panopticThigh,axis=1)))

        # original_vis = conv_visGl_form(normalizedPose)
        # #showSkeleton([original_vis])    #Debug

        rest, names, _ = BVH.load('./cmu/rest.bvh')

        anim = rest.copy()
        anim.positions = anim.positions.repeat(normalizedPose.shape[0], axis=0)
        anim.rotations.qs = anim.rotations.qs.repeat(normalizedPose.shape[0], axis=0)

        cmuMocapJoints = Animation.positions_global(anim)
        # cmuMocapHeight = max(cmuMocapJoints[:,:,1].flatten()) -   min(cmuMocapJoints[:,:,1].flatten())
        # cmuMocapHeight = cmuMocapHeight*1.1

        cmuThigh = cmuMocapJoints[:,2,:] - cmuMocapJoints[:,3,:]
        cmuThigh = cmuThigh**2
        cmuMocapHeight = np.mean(np.sqrt(np.sum(cmuThigh,axis=1)))
        cmuMocapHeight = cmuMocapHeight*0.9
        

        scaleRatio = cmuMocapHeight/panopticHeight
        print("cmuMocapHeight: %f, panopticHeight %f, scaleRatio: %f " % (cmuMocapHeight,panopticHeight,scaleRatio) )
        normalizedPose = normalizedPose * scaleRatio   #rescaling
        original_vis = conv_visGl_form(normalizedPose)

        """Compute and Set Root orientation"""
        across1 = normalizedPose[:,3] - normalizedPose[:,9] #Right -> left (3)  Shoulder
        across0 = normalizedPose[:,6] - normalizedPose[:, 12] #Right -> left (6) Hips
        across = across0 + across1 #frame x 3
        across = across / np.sqrt((across**2).sum(axis=-1))[...,np.newaxis] ##frame x 3. Unit vectors

        forward = np.cross(across, np.array([[0,-1,0]]))
        forward = forward / np.sqrt((forward**2).sum(axis=-1))[...,np.newaxis]
        target = np.array([[0,0,1]]).repeat(len(forward), axis=0)

        anim.positions[:,0] = normalizedPose[:,2] + np.array([0.0, 2.4, 0.0])  #Set root's movement by hipCenter joints (idx=2)
        #anim.positions[:,0] = cuttargets[:,0] + np.array([0.0, 1.4, 0.0])
        anim.rotations[:,0:1] = -Quaternions.between(forward, target)[:,np.newaxis]   #0:1 means root

        # # # #Debug
        # targets = Animation.positions_global(anim)
        # targets_vis = conv_visGl_form(targets)
        # showSkeleton([original_vis*5,targets_vis*5])    #Debug

        #Note: we flip Y axis for the Panoptic, so left-right are flipped
        mapping = {
                13:0, 16:1,#12:12, 15:13, 16:15,
                2:12, 3:13,  4:14, #left hip,knee, ankle, footend
                7:6, 8:7,  9:8, #right hip,knee, ankle, footend
                17:0, 18:9, 19:10, 20:11, 20:11,
                24:0, 25:3, 26:4, 27:5, 27:5
                #17:13, 18:25, 19:26, 20:27, 22:30,
                #24:12, 25:17, 26:18, 27:19, 29:22
        }

        targetmap = {}
        #targetmap[1] = anim.positions[:,0,:]
        for k in mapping:
            #anim.rotations[:,k] = stanim.rotations[:,mapping[k]]
            targetmap[k] = normalizedPose[:,mapping[k],:]

        #ik = JacobianInverseKinematics(anim, targetmap, iterations=10, damping=10.0, silent=True)
        ik = JacobianInverseKinematics(anim,targetmap , iterations=20, damping=10.0, silent=True)
        ik()

        # """Debug"""
        # targets = Animation.positions_global(anim)
        # targets_vis = conv_visGl_form(targets)
        # showSkeleton([targets_vis*5,original_vis*5])    #Debug

        BVH.save(outputFolder + seqName + '_p' + str(pid)+ '.bvh', anim, names)
        
