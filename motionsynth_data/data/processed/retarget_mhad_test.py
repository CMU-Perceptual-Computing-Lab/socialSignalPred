import os
import sys
import numpy as np

sys.path.append('../../motion')
import BVH as BVH
import Animation as Animation
from Quaternions import Quaternions
from Pivots import Pivots
from InverseKinematics import JacobianInverseKinematics

#by jhugestar
sys.path.append('/ssd/codes/glvis_python/')
from Visualize_human_gl import showSkeleton #opengl visualization 

def conv_visGl_form(rest_targets): #skel: (frames, 31, 3)

    rest_targets = rest_targets.reshape(rest_targets.shape[0],rest_targets.shape[1]*rest_targets.shape[2]) #(frames, 93)
    rest_targets = np.swapaxes(rest_targets,0,1)  #(93,frames)

    return rest_targets


#path='../external/MHAD/BerkeleyMHAD/Mocap/SkeletalData'
path='/ssd/data/berkeleyMHAD/BerkeleyMHAD/Mocap/SkeletalData'

bvh_files = [path+'/'+f for f in sorted(list(os.listdir(path)))
    if os.path.isfile(os.path.join(path,f)) and f.endswith('.bvh')]

database = bvh_files

rest, names, _ = BVH.load('./cmu/rest.bvh')
#BVH.save('./mhad/rest.bvh', rest, names)

for i, filename in enumerate(database):
    
    if i <20:
        continue
    print('%i of %i Processing %s' % (i+1, len(database), filename))
    
    mhanim, mhnames, ftime = BVH.load(filename) #mhanim: (frames,30)
    mhanim = mhanim[::4] #Every 4th frames. E.g., (2797,30) -> (700,30)
    mhanim.positions = mhanim.positions * 0.19
    mhanim.offsets = mhanim.offsets * 0.19
    
    #Debug
    targets = Animation.positions_global(mhanim)
    targets_vis = conv_visGl_form(targets)
    #showSkeleton([targets_vis])
    
    anim = rest.copy()
    anim.positions = anim.positions.repeat(len(targets), axis=0)
    anim.rotations.qs = anim.rotations.qs.repeat(len(targets), axis=0)
    
    anim.positions[:,0] = targets[:,1]
    
    # mapping = {
    #      0:  0,
    #      2: 24,  3: 26,  4: 28,  5: 29,
    #      7: 18,  8: 20,  9: 22, 10: 23,
    #     11:  1, 12:  2, 13:  3, 15:  4, 16: 5,
    #     18: 13, 19: 15, 20: 17,
    #     25:  7, 26:  9, 27: 11}
    # # mapping = {
    # #       0:  0}
    
    # targetmap = {}
    
    # for k in mapping:
    #     targetmap[k] = targets[:,mapping[k]]
    
    # ik = JacobianInverseKinematics(anim, targetmap, iterations=10, damping=2.0, silent=True)
    # ik()
    

    #Debug
    targets_retarget = Animation.positions_global(anim)
    targets_retarget_vis_ = conv_visGl_form(targets_retarget)
    showSkeleton([targets_vis*10,targets_retarget_vis_*10])
    #BVH.save('./mhad/'+os.path.split(filename)[1], anim, names, 1.0/120)
    
