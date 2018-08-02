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

#by jhugestar
sys.path.append('/ssd/codes/glvis_python/')
from Visualize_human_gl import showSkeleton #opengl visualization 


def conv_visGl_form(rest_targets): #skel: (frames, 31, 3)

    rest_targets = rest_targets.reshape(rest_targets.shape[0],rest_targets.shape[1]*rest_targets.shape[2]) #(frames, 93)
    rest_targets = np.swapaxes(rest_targets,0,1)  #(93,frames)

    return rest_targets



database = []

#dbroot = '../external/cmu/'
dbroot = '/ssd/data/cmu-mocap/'

#info = open(dbroot+'cmu-mocap-index-text.txt', 'r', errors='ignore')
info = open(dbroot+'cmu-mocap-index-text.txt', 'r')

for line in info.readlines()[16:]:
    if line.strip() == '': continue
    m0 = re.match('Subject #\d+ \(([^)]*)\)\n', line)
    if m0:
        continue
        
    m1 = re.match('(\d+_\d+)\s+([^\n]*)\n', line)
    if m1:
        id0, id1 = m1.group(1).split('_')
        database.append((id0, id1))
    
info.close()

baddata = set([('90', '10')])

database = [data for data in database 
    if (data[0], data[1]) not in baddata]

""" Begin Processing """

rest, names, _ = BVH.load(dbroot+'/data/001/01_01.bvh')
rest_targets = Animation.positions_global(rest) #output: (frames, jointNum (31), 3)

# Visualizing CMU mocap. re-arrangement for visualization
# rest_targets = rest_targets.reshape(rest_targets.shape[0],rest_targets.shape[1]*rest_targets.shape[2]) #(frames, 93)
# rest_targets = np.swapaxes(rest_targets,0,1)  #(93,frames)
# showSkeleton([rest_targets*10])  #(skelNum, dim, frames)

rest_height = rest_targets[0,:,1].max() #First frame, max Y value

skel = rest.copy() #this one has all frames
skel.positions = rest.positions[0:1] #rest.positions: (frames, 31, 3).  skel.positions has (1,31,3)
skel.rotations = rest.rotations[0:1] #rest.rotations: (frames, 31). Each instance is Quaternions. skel.rotations has (1,31)
skel.positions[:,0,0] = 0   #root pose x==0
skel.positions[:,0,2] = 0   #root pose z==0.. So only Root Y axis is valid
skel.offsets[0,0] = 0  #skel.offsets: (31,3)
skel.offsets[0,2] = 0   #skel.offsets: (31,3)

# # # Visualizing CMU mocap. re-arrangement for visualization
# rest_targets = Animation.positions_global(skel) #output: (frames, jointNum (31), 3)
# rest_targets = rest_targets.reshape(rest_targets.shape[0],rest_targets.shape[1]*rest_targets.shape[2]) #(frames, 93)
# rest_targets = np.swapaxes(rest_targets,0,1)  #(93,frames)

#scaling skeletons
skel.offsets = skel.offsets * 6.25 
skel.positions = skel.positions * 6.25
rest_targets = Animation.positions_global(skel) #output: (frames, jointNum (31), 3)
rest_height_scaled = rest_targets[0,:,1].max() #First frame, max Y value

print('height {0} -> {1}'.format(rest_height,rest_height_scaled)) #height 25.4452109527 -> 159.032568455

# # Visualizing CMU mocap. re-arrangement for visualization
# rest_targets_scaled = Animation.positions_global(skel) #output: (frames, jointNum (31), 3)
# rest_targets_scaled = rest_targets_scaled.reshape(rest_targets_scaled.shape[0],rest_targets_scaled.shape[1]*rest_targets_scaled.shape[2]) #(frames, 93)
# rest_targets_scaled = np.swapaxes(rest_targets_scaled,0,1)  #(93,frames)
# showSkeleton([rest_targets,rest_targets_scaled])  #(skelNum, dim, frames)


#BVH.save('./skel_motionbuilder.bvh', skel, names)   #This file has scaled version (6.75 bigger)

rest.positions = rest.offsets[np.newaxis] #rest.offsets: (31,3),  rest.positions:(1,31,3)
rest.rotations.qs = rest.orients.qs[np.newaxis] #rest.orients.qs: (31,4), rest.rotations.qs: (1,31,4)

#BVH.save('./cmu/rest.bvh', rest, names)     #This file has origial scale (rest) with root on the origin (0,0,). Has only single frame

for i, data in enumerate(database):

    if i<5:
        continue

    #filename = dbroot+data[0]+'/'+data[0]+'_'+data[1]+'.bvh'
    filename = dbroot+'data/'+data[0].zfill(3)+'/'+data[0]+'_'+data[1]+'.bvh'
    
    print('%i of %i Processing %s' % (i+1, len(database), filename))
    
    anim, _, ftime = BVH.load(filename)
    anim_targets = Animation.positions_global(anim)  #anim_targets has 1 more frame than anim
    anim_height = anim_targets[0,:,1].max()
    print(anim_height)

    
    #skel_vis_init = conv_visGl_form(anim_targets) #debug
    #showSkeleton([skel_vis_init])#debug

    targets = (rest_height / anim_height) * anim_targets[1:] #make motion data to "rest_height" scale....what about Jump?
                                                             #Drop the initial frame to have the same frame as anim. 
                                                             #targets: (frames,31,3)

    skel_vis_scaled = conv_visGl_form(targets) #debug
    #showSkeleton([skel_vis_scaled])#debug
    
    

    anim = anim[1:]
    anim.orients.qs = rest.orients.qs.copy()    #rest.orients.qs: (31,4)
    anim.offsets = rest.offsets.copy()      #Copy base skeleton (joint orientation and oofest)
    anim.positions[:,0] = targets[:,0]    #copy the root position
    anim.positions[:,1:] = rest.positions[:,1:].repeat(len(targets), axis=0) #rest.positions[:,1:] : (1,30,3)
                                                                             #Copy the rest's relative joint positions (same as offset of default pose) for all frames
    ### Here anim has the same skeleton structure (orient, offest, position) as the default rest pose 
    ### Now, only need to convert "rotations" and "parents" variables 

    targetmap = {}  #targetmap[jointIdx] = np.array with (frames,3) shape. Here jointIdx = 0~30 
    
    #for ti in range(0,1):#range(targets.shape[1]):  #ti: 0-30
    for ti in range(targets.shape[1]):  #ti: 0-30
        targetmap[ti] = targets[:,ti]  #targets[:,ti]: (frames, 3)
        
    
    ik = JacobianInverseKinematics(anim, targetmap, iterations=10, damping=2.0, silent=False)
    ik()

    #debug
    anim_targets_retarget = Animation.positions_global(anim)
    skel_vis_retarget = conv_visGl_form(anim_targets_retarget) #debug
    showSkeleton([skel_vis_scaled*3, skel_vis_retarget*3])#debug
    
    
    
    #BVH.save('./cmu/'+data[0]+'_'+data[1]+'.bvh', anim, names, ftime)    

