import os
import sys
import numpy as np
import scipy.interpolate as interpolate

sys.path.append('../../motion')
import BVH
import Animation
from Quaternions import Quaternions
from InverseKinematics import BasicJacobianIK, JacobianInverseKinematics


#by jhugestar
sys.path.append('/ssd/codes/glvis_python/')
from Visualize_human_gl import showSkeleton #opengl visualization 


def conv_visGl_form(rest_targets): #skel: (frames, 31, 3)
    rest_targets = rest_targets.reshape(rest_targets.shape[0],rest_targets.shape[1]*rest_targets.shape[2]) #(frames, 93)
    rest_targets = np.swapaxes(rest_targets,0,1)  #(93,frames)

    return rest_targets



path='../external/edin_terrain/'

bvh_files = [
    path+f for f in sorted(list(os.listdir(path)))
    if os.path.isfile(os.path.join(path,f))
    and f.endswith('.bvh')]

rest, names, _ = BVH.load('./cmu/rest.bvh')
BVH.save('./edin_terrain/rest.bvh', rest, names)

for i, bvh_file in enumerate(bvh_files):
    
    print('%i of %i Processing %s' % (i+1, len(bvh_files), bvh_file))
    
    bvhanim, bvhnames, _ = BVH.load(bvh_file)
    bvhanim = bvhanim[1:]
    bvhanim.positions = bvhanim.positions * 0.15
    bvhanim.offsets = bvhanim.offsets * 0.15

    targets = Animation.positions_global(bvhanim)

    #debug
    skel_vis_origin = conv_visGl_form(targets) #debug
    #showSkeleton([skel_vis*3])#debug
    
    cuts = [(None, None)]
    
    if   'Dan_walking_up_04' in bvh_file:
        cuts = [(None, 7310), (7420, None)]
    
    for ci, cut in enumerate(cuts):
    
        cuttargets = targets[cut[0]:cut[1]]
    
        """Debug"""
        cuttargets = cuttargets[:1000]

        anim = rest.copy()
        anim.positions = anim.positions.repeat(len(cuttargets), axis=0)
        anim.rotations.qs = anim.rotations.qs.repeat(len(cuttargets), axis=0)
        

        """Testing code"""
        #anim.positions[:,0] = cuttargets[:,0] + np.array([0.0, 1.4, 0.0])

        """Original code"""
        across1 = cuttargets[:,11] - cuttargets[:,16] #Right -> left (11)  Shoulders
        across0 = cuttargets[:, 1] - cuttargets[:, 5] #Right -> left (1) Hips
        across = across0 + across1 #frame x 3
        across = across / np.sqrt((across**2).sum(axis=-1))[...,np.newaxis] ##frame x 3. Unit vectors

        forward = np.cross(across, np.array([[0,1,0]]))
        forward = forward / np.sqrt((forward**2).sum(axis=-1))[...,np.newaxis]
        target = np.array([[0,0,1]]).repeat(len(forward), axis=0)

        anim.positions[:,0] = cuttargets[:,0] + np.array([0.0, 1.4, 0.0])
        anim.rotations[:,0:1] = -Quaternions.between(forward, target)[:,np.newaxis]   #0:1 means root
        
        mapping = {
              2:  1,  3:  2,  4:  3,  5: 4,
              7:  5,  8:  6,  9:  7, 10: 8,
             12:  9, 16: 20,
             17: 10, 18: 11, 19: 12, 20: 13, 22: 14,
             24:  9, 25: 16, 26: 17, 27: 18, 29: 19}

        # mapping = {
        #       12:  9, 14: 9
        # }
        
        targetmap = {}
        
        for k in mapping:
            targetmap[k] = cuttargets[:,mapping[k]]
        
        ik = JacobianInverseKinematics(anim, targetmap, iterations=10, damping=2.0, silent=False)
        ik()
        
        #debug
        targets = Animation.positions_global(anim)
        skel_vis_new = conv_visGl_form(targets) #debug
        showSkeleton([skel_vis_origin*3,skel_vis_new*3])#debug
    
        #BVH.save('./edin_terrain/'+os.path.split(bvh_file)[1].replace('.bvh', '_%03i.bvh' % ci), anim, names, 1.0/120)
    
    
