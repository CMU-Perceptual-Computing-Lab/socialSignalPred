import re
import sys
import numpy as np
import scipy.io as io

sys.path.append('../../motion')
import BVH as BVH
import Animation as Animation
from Quaternions import Quaternions
from Pivots import Pivots
from InverseKinematics import BasicJacobianIK, JacobianInverseKinematics

import os
os.environ["CDF_LIB"] = "/home/hanbyulj/cdf36_4-dist/lib"
from spacepy import pycdf


#by jhugestar
sys.path.append('/ssd/codes/glvis_python/')
from Visualize_human_gl import showSkeleton #opengl visualization 

def conv_visGl_form(rest_targets): #skel: (frames, 31, 3)

    rest_targets = rest_targets.reshape(rest_targets.shape[0],rest_targets.shape[1]*rest_targets.shape[2]) #(frames, 93)
    rest_targets = np.swapaxes(rest_targets,0,1)  #(93,frames)

    return rest_targets


database = []

dbroot = '/ssd/data/h36m/'

sourcePath='/media/hanbyulj/ssd/data/h36m/'
#subjectIdx = 5
subjects = [1,5, 6, 7, 8, 9, 11]

for subjectIdx in subjects:
    path = "{0}/S{1}/MyPoseFeatures/D3_Positions_mono_universal/".format(sourcePath,subjectIdx)
    cdf_files = [
        path+f for f in sorted(list(os.listdir(path)))
        if os.path.isfile(os.path.join(path,f))
        and f.endswith('.cdf')]

    for i, cdf_file in enumerate(cdf_files):
        
        print('%i of %i Processing %s' % (i+1, len(cdf_file), cdf_file))

        cdf = pycdf.CDF(cdf_file)
        #cdf = pycdf.CDF('/media/hanbyulj/ssd/data/h36m/S1/MyPoseFeatures/D3_Positions_mono_universal/Directions.54138969.cdf')
        #cdf = pycdf.CDF('/media/hanbyulj/ssd/data/h36m/S5/MyPoseFeatures/D3_Positions_mono_universal/Directions 1.54138969.cdf')
        #cdf = pycdf.CDF('/media/hanbyulj/ssd/data/h36m/S7/MyPoseFeatures/D3_Positions_mono_universal/Directions.54138969.cdf')
        normalizedPose = cdf['Pose'][0,:,:]

        """Debug: cropping to short sequences """
        normalizedPose = normalizedPose[:500,:] #For debugging

        normalizedPose = np.reshape(normalizedPose,(normalizedPose.shape[0],-1,3)) # (frames,32,3)
        initPose = normalizedPose[:1,:1,:]

        #Set foot end on the ground
        heightOffset = max(normalizedPose[0,5,1],normalizedPose[0,10,1])+50
        initPose[0,0,1] = heightOffset #To keep y axis

        initPose = np.repeat(initPose,normalizedPose.shape[0],axis=0)
        initPose = np.repeat(initPose,normalizedPose.shape[1],axis=1) # normalizedPose.shape[1]==32
        normalizedPose = normalizedPose- initPose  #(frame,32,3)
        normalizedPose = normalizedPose * 0.05

        normalizedPose = normalizedPose * 0.33
        #normalizedPose = np.reshape(normalizedPose,(normalizedPose.shape[0],-1))

        #Flip Y axis
        normalizedPose[:,:,1] = -normalizedPose[:,:,1]


        original_vis = conv_visGl_form(normalizedPose)*5
        #showSkeleton([original_vis])    #Debug

        rest, names, _ = BVH.load('./cmu/rest.bvh')

        anim = rest.copy()
        anim.positions = anim.positions.repeat(normalizedPose.shape[0], axis=0)
        anim.rotations.qs = anim.rotations.qs.repeat(normalizedPose.shape[0], axis=0)


        """Compute and Set Root orientation"""
        across1 = normalizedPose[:,25] - normalizedPose[:,17] #Right -> left (25)  Shoulder
        across0 = normalizedPose[:,1] - normalizedPose[:, 6] #Right -> left (2) Hips
        across = across0 + across1 #frame x 3
        across = across / np.sqrt((across**2).sum(axis=-1))[...,np.newaxis] ##frame x 3. Unit vectors

        forward = np.cross(across, np.array([[0,1,0]]))
        forward = forward / np.sqrt((forward**2).sum(axis=-1))[...,np.newaxis]
        target = np.array([[0,0,1]]).repeat(len(forward), axis=0)

        #anim.positions[:,0] = normalizedPose[:,0]   #Set root's movement
        #anim.positions[:,0,1]  = anim.positions[:,0,1]+2.0
        anim.positions[:,0] = normalizedPose[:,0] + np.array([0.0, 2.0, 0.0])   #Set root's movement
        #anim.positions[:,0] = cuttargets[:,0] + np.array([0.0, 1.4, 0.0])
        anim.rotations[:,0:1] = -Quaternions.between(forward, target)[:,np.newaxis]   #0:1 means root


        # #Debug
        # targets = Animation.positions_global(anim)
        # targets_vis = conv_visGl_form(targets)*5
        # showSkeleton([original_vis,targets_vis])    #Debug

        mapping = {
                12:12, 15:13, 16:15,
                2:1, 3:2,  4:3, 5:4, #left hip,knee, ankle, footend
                7:6, 8:7,  9:8, 10:9, #right hip,knee, ankle, footend
                17:13, 18:25, 19:26, 20:27, 22:30,
                24:12, 25:17, 26:18, 27:19, 29:22
        }

        targetmap = {}
        #targetmap[1] = anim.positions[:,0,:]
        for k in mapping:
            #anim.rotations[:,k] = stanim.rotations[:,mapping[k]]
            targetmap[k] = normalizedPose[:,mapping[k],:]

        #ik = JacobianInverseKinematics(anim, targetmap, iterations=10, damping=10.0, silent=True)
        ik = JacobianInverseKinematics(anim,targetmap , iterations=20, damping=10.0, silent=True)
        ik()

        """Debug"""
        targets = Animation.positions_global(anim)
        targets_vis = conv_visGl_form(targets)*5
        showSkeleton([targets_vis,original_vis])    #Debug

        #BVH.save('./h36m/'+'S'+str(subjectIdx) + '-' + os.path.split(cdf_file)[1].replace('.cdf', '.bvh'), anim, names)
        
