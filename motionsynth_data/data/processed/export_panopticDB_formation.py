import os
import sys
import numpy as np
import scipy.io as io
import scipy.ndimage.filters as filters

sys.path.append('../../motion')
import BVH as BVH
import Animation as Animation
from Quaternions import Quaternions
from Pivots import Pivots

import cPickle as pickle

from sklearn.preprocessing import normalize

def softmax(x, **kw):
    softness = kw.pop('softness', 1.0)
    maxi, mini = np.max(x, **kw), np.min(x, **kw)
    return maxi + np.log(softness + np.exp(mini - maxi))

def softmin(x, **kw):
    return -softmax(-x, **kw)
    
def process_file(filename, window=1, window_step=1):
    
    anim, names, frametime = BVH.load(filename)
    
    # """ Convert to 60 fps """
    # anim = anim[::2]
    # Origianl Human3.6 has 50 fps. So, no smapling is done
    
    """ Do FK """
    global_positions = Animation.positions_global(anim)
    
    """ Remove Uneeded Joints """
    positions = global_positions[:,np.array([
         0,
         2,  3,  4,  5,
         7,  8,  9, 10,
        12, 13, 15, 16,
        18, 19, 20, 22,
        25, 26, 27, 29])]
    
    """ Put on Floor """
    fid_l, fid_r = np.array([4,5]), np.array([8,9])
    foot_heights = np.minimum(positions[:,fid_l,1], positions[:,fid_r,1]).min(axis=1)
    floor_height = softmin(foot_heights, softness=0.5, axis=0)
    
    positions[:,:,1] -= floor_height

    """ Add Reference Joint """
    trajectory_filterwidth = 3
    reference = positions[:,0] * np.array([1,0,1])
    reference = filters.gaussian_filter1d(reference, trajectory_filterwidth, axis=0, mode='nearest')    
    positions = np.concatenate([reference[:,np.newaxis], positions], axis=1)
    
    """ Get Foot Contacts """
    velfactor, heightfactor = np.array([0.05,0.05]), np.array([3.0, 2.0])
    
    feet_l_x = (positions[1:,fid_l,0] - positions[:-1,fid_l,0])**2
    feet_l_y = (positions[1:,fid_l,1] - positions[:-1,fid_l,1])**2
    feet_l_z = (positions[1:,fid_l,2] - positions[:-1,fid_l,2])**2
    feet_l_h = positions[:-1,fid_l,1]
    feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor) & (feet_l_h < heightfactor)).astype(np.float)
    
    feet_r_x = (positions[1:,fid_r,0] - positions[:-1,fid_r,0])**2
    feet_r_y = (positions[1:,fid_r,1] - positions[:-1,fid_r,1])**2
    feet_r_z = (positions[1:,fid_r,2] - positions[:-1,fid_r,2])**2
    feet_r_h = positions[:-1,fid_r,1]
    feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor) & (feet_r_h < heightfactor)).astype(np.float)
    
    """ Get Root Velocity """
    velocity = (positions[1:,0:1] - positions[:-1,0:1]).copy()
    
    """ Remove Translation """
    positions[:,:,0] = positions[:,:,0] - positions[:,0:1,0]
    positions[:,:,2] = positions[:,:,2] - positions[:,0:1,2]
    
    """ Get Forward Direction """
    sdr_l, sdr_r, hip_l, hip_r = 14, 18, 2, 6
    across1 = positions[:,hip_l] - positions[:,hip_r]
    across0 = positions[:,sdr_l] - positions[:,sdr_r]
    across = across0 + across1
    across = across / np.sqrt((across**2).sum(axis=-1))[...,np.newaxis]
    
    direction_filterwidth = 20
    forward = np.cross(across, np.array([[0,1,0]]))
    forward = filters.gaussian_filter1d(forward, direction_filterwidth, axis=0, mode='nearest')    
    forward = forward / np.sqrt((forward**2).sum(axis=-1))[...,np.newaxis]

    """ Remove Y Rotation """
    target = np.array([[0,0,1]]).repeat(len(forward), axis=0)
    rotation = Quaternions.between(forward, target)[:,np.newaxis]    
    positions = rotation * positions
    
    """ Get Root Rotation """
    velocity = rotation[1:] * velocity
    rvelocity = Pivots.from_quaternions(rotation[1:] * -rotation[:-1]).ps
    
    """ Add Velocity, RVelocity, Foot Contacts to vector """
    positions = positions[:-1]
    positions = positions.reshape(len(positions), -1)
    positions = np.concatenate([positions, velocity[:,:,0]], axis=-1)
    positions = np.concatenate([positions, velocity[:,:,2]], axis=-1)
    positions = np.concatenate([positions, rvelocity], axis=-1)
    positions = np.concatenate([positions, feet_l, feet_r], axis=-1)
        
    """ Slide over windows """
    windows = []
    windows_classes = []
    
    for j in range(0, len(positions)-window//8, window_step):
    
        """ If slice too small pad out by repeating start and end poses """
        slice = positions[j:j+window]
        if len(slice) < window:
            left  = slice[:1].repeat((window-len(slice))//2 + (window-len(slice))%2, axis=0)
            left[:,-7:-4] = 0.0
            right = slice[-1:].repeat((window-len(slice))//2, axis=0)
            right[:,-7:-4] = 0.0
            slice = np.concatenate([left, slice, right], axis=0)
        
        if len(slice) != window: raise Exception()
        
        windows.append(slice)
        
        """ Find Class """
        cls = -1
        if filename.startswith('hdm05'):
            cls_name = os.path.splitext(os.path.split(filename)[1])[0][7:-8]
            cls = class_names.index(class_map[cls_name]) if cls_name in class_map else -1
        if filename.startswith('styletransfer'):
            cls_name = os.path.splitext(os.path.split(filename)[1])[0]
            cls = np.array([
                styletransfer_motions.index('_'.join(cls_name.split('_')[1:-1])),
                styletransfer_styles.index(cls_name.split('_')[0])])
        windows_classes.append(cls)
        
    return windows, windows_classes





def process_file_withSpeech(filename, window=240, window_step=120):
    
    anim, names, frametime = BVH.load(filename)

    #Load speech info
    seqName = os.path.basename(filename)
    speech_fileName = seqName[:-7] + '.pkl'
    speechPath = './panopticDB_pkl_speech_hagglingProcessed/' +speech_fileName
    speechData = pickle.load( open( speechPath, "rb" ) )

    if '_p0.bvh' in filename:
        speechData = speechData['speechData'][0]
    elif '_p1.bvh' in filename:
        speechData = speechData['speechData'][1]
    elif '_p2.bvh' in filename:
        speechData = speechData['speechData'][2]
    
    # """ Convert to 60 fps """
    # anim = anim[::2]
    # Origianl Human3.6 has 50 fps. So, no smapling is done
    
    """ Do FK """
    global_positions = Animation.positions_global(anim)
    
    """ Remove Uneeded Joints """
    positions = global_positions[:,np.array([
         0,
         2,  3,  4,  5,
         7,  8,  9, 10,
        12, 13, 15, 16,
        18, 19, 20, 22,
        25, 26, 27, 29])]
    
    """ Put on Floor """
    fid_l, fid_r = np.array([4,5]), np.array([8,9])
    foot_heights = np.minimum(positions[:,fid_l,1], positions[:,fid_r,1]).min(axis=1)
    floor_height = softmin(foot_heights, softness=0.5, axis=0)
    
    positions[:,:,1] -= floor_height

    """ Add Reference Joint """
    trajectory_filterwidth = 3
    reference = positions[:,0] * np.array([1,0,1])
    reference = filters.gaussian_filter1d(reference, trajectory_filterwidth, axis=0, mode='nearest')    
    positions = np.concatenate([reference[:,np.newaxis], positions], axis=1)
    
    """ Get Foot Contacts """
    velfactor, heightfactor = np.array([0.05,0.05]), np.array([3.0, 2.0])
    
    feet_l_x = (positions[1:,fid_l,0] - positions[:-1,fid_l,0])**2
    feet_l_y = (positions[1:,fid_l,1] - positions[:-1,fid_l,1])**2
    feet_l_z = (positions[1:,fid_l,2] - positions[:-1,fid_l,2])**2
    feet_l_h = positions[:-1,fid_l,1]
    feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor) & (feet_l_h < heightfactor)).astype(np.float)
    
    feet_r_x = (positions[1:,fid_r,0] - positions[:-1,fid_r,0])**2
    feet_r_y = (positions[1:,fid_r,1] - positions[:-1,fid_r,1])**2
    feet_r_z = (positions[1:,fid_r,2] - positions[:-1,fid_r,2])**2
    feet_r_h = positions[:-1,fid_r,1]
    feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor) & (feet_r_h < heightfactor)).astype(np.float)
    
    """ Get Root Velocity """
    velocity = (positions[1:,0:1] - positions[:-1,0:1]).copy()
    
    """ Remove Translation """
    positions[:,:,0] = positions[:,:,0] - positions[:,0:1,0]
    positions[:,:,2] = positions[:,:,2] - positions[:,0:1,2]
    
    """ Get Forward Direction """
    sdr_l, sdr_r, hip_l, hip_r = 14, 18, 2, 6
    across1 = positions[:,hip_l] - positions[:,hip_r]
    across0 = positions[:,sdr_l] - positions[:,sdr_r]
    across = across0 + across1
    across = across / np.sqrt((across**2).sum(axis=-1))[...,np.newaxis]
    
    direction_filterwidth = 20
    forward = np.cross(across, np.array([[0,1,0]]))
    forward = filters.gaussian_filter1d(forward, direction_filterwidth, axis=0, mode='nearest')    
    forward = forward / np.sqrt((forward**2).sum(axis=-1))[...,np.newaxis]

    """ Remove Y Rotation """
    target = np.array([[0,0,1]]).repeat(len(forward), axis=0)
    rotation = Quaternions.between(forward, target)[:,np.newaxis]    
    positions = rotation * positions
    
    """ Get Root Rotation """
    velocity = rotation[1:] * velocity
    rvelocity = Pivots.from_quaternions(rotation[1:] * -rotation[:-1]).ps
    
    """ Add Velocity, RVelocity, Foot Contacts to vector """
    positions = positions[:-1]
    positions = positions.reshape(len(positions), -1)
    positions = np.concatenate([positions, velocity[:,:,0]], axis=-1)
    positions = np.concatenate([positions, velocity[:,:,2]], axis=-1)
    positions = np.concatenate([positions, rvelocity], axis=-1)
    positions = np.concatenate([positions, feet_l, feet_r], axis=-1)
        
    """ Slide over windows """
    windows = []
    windows_classes = []
    

    print("skelSize {0} vs speechSize {1}".format(positions.shape[0],speechData['indicator'].shape[0]))
    for j in range(0, len(positions)-window//8, window_step):
    
        """ If slice too small pad out by repeating start and end poses """
        slice = positions[j:j+window]
        if len(slice) < window:
            break
            # left  = slice[:1].repeat((window-len(slice))//2 + (window-len(slice))%2, axis=0)
            # left[:,-7:-4] = 0.0
            # right = slice[-1:].repeat((window-len(slice))//2, axis=0)
            # right[:,-7:-4] = 0.0
            # slice = np.concatenate([left, slice, right], axis=0)
        
        if len(slice) != window: raise Exception()
        
        windows.append(slice)
        
        # """ Find Class """
        # cls = -1
        # if filename.startswith('hdm05'):
        #     cls_name = os.path.splitext(os.path.split(filename)[1])[0][7:-8]
        #     cls = class_names.index(class_map[cls_name]) if cls_name in class_map else -1
        # if filename.startswith('styletransfer'):
        #     cls_name = os.path.splitext(os.path.split(filename)[1])[0]
        #     cls = np.array([
        #         styletransfer_motions.index('_'.join(cls_name.split('_')[1:-1])),
        #         styletransfer_styles.index(cls_name.split('_')[0])])
        cls = speechData['indicator'][j:j+window]
        windows_classes.append(cls)
        
    return windows, windows_classes

"""
    Processing skeleton information with speech
    output::
    windows: list with 3 element (buyer, winner, loser) 
    windows_speech: list with 3 element (buyer, winner, loser) 

    windows and windows_speech should have the same size
"""
def old__process_file_withSpeech_byGroup(filename, window=240, window_step=120):
    
    if not '_p0.bvh' in filename:
        return

    #Load speech info
    seqName = os.path.basename(filename)
    speech_fileName = seqName[:-7] + '.pkl'
    speechPath = './panopticDB_pkl_speech_hagglingProcessed/' +speech_fileName
    speechData_raw = pickle.load( open( speechPath, "rb" ) )

    positions_list=list()
    speechData_list=list()
    
    for pIdx in range(0,3):
        """ Do FK """
        if pIdx==0:
            anim, names, frametime = BVH.load(filename)
        else:
            newFileName = filename.replace('_p0.bvh','_p{0}.bvh'.format(pIdx));
            anim, names, frametime = BVH.load(newFileName)

        if pIdx==0:#_p0.bvh' in filename:
            speechData = speechData_raw['speechData'][0]
        elif pIdx==1:#'_p1.bvh' in filename:
            speechData = speechData_raw['speechData'][1]
        elif pIdx==2:#'_p2.bvh' in filename:
            speechData = speechData_raw['speechData'][2]
        
        

        # """ Convert to 60 fps """
        # anim = anim[::2]
        # Origianl Human3.6 has 50 fps. So, no smapling is done
        
        """ Do FK """
        global_positions = Animation.positions_global(anim)
        
        """ Remove Uneeded Joints """
        positions = global_positions[:,np.array([
            0,
            2,  3,  4,  5,
            7,  8,  9, 10,
            12, 13, 15, 16,
            18, 19, 20, 22,
            25, 26, 27, 29])]
        
        """ Put on Floor """
        fid_l, fid_r = np.array([4,5]), np.array([8,9])
        foot_heights = np.minimum(positions[:,fid_l,1], positions[:,fid_r,1]).min(axis=1)
        floor_height = softmin(foot_heights, softness=0.5, axis=0)
        
        positions[:,:,1] -= floor_height

        """ Add Reference Joint """
        trajectory_filterwidth = 3
        reference = positions[:,0] * np.array([1,0,1])
        reference = filters.gaussian_filter1d(reference, trajectory_filterwidth, axis=0, mode='nearest')    
        positions = np.concatenate([reference[:,np.newaxis], positions], axis=1)
        
        """ Get Foot Contacts """
        velfactor, heightfactor = np.array([0.05,0.05]), np.array([3.0, 2.0])
        
        feet_l_x = (positions[1:,fid_l,0] - positions[:-1,fid_l,0])**2
        feet_l_y = (positions[1:,fid_l,1] - positions[:-1,fid_l,1])**2
        feet_l_z = (positions[1:,fid_l,2] - positions[:-1,fid_l,2])**2
        feet_l_h = positions[:-1,fid_l,1]
        feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor) & (feet_l_h < heightfactor)).astype(np.float)
        
        feet_r_x = (positions[1:,fid_r,0] - positions[:-1,fid_r,0])**2
        feet_r_y = (positions[1:,fid_r,1] - positions[:-1,fid_r,1])**2
        feet_r_z = (positions[1:,fid_r,2] - positions[:-1,fid_r,2])**2
        feet_r_h = positions[:-1,fid_r,1]
        feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor) & (feet_r_h < heightfactor)).astype(np.float)
        
        """ Get Root Velocity """
        velocity = (positions[1:,0:1] - positions[:-1,0:1]).copy()
        
        """ Remove Translation """
        positions[:,:,0] = positions[:,:,0] - positions[:,0:1,0]
        positions[:,:,2] = positions[:,:,2] - positions[:,0:1,2]
        
        """ Get Forward Direction """
        sdr_l, sdr_r, hip_l, hip_r = 14, 18, 2, 6
        across1 = positions[:,hip_l] - positions[:,hip_r]
        across0 = positions[:,sdr_l] - positions[:,sdr_r]
        across = across0 + across1
        across = across / np.sqrt((across**2).sum(axis=-1))[...,np.newaxis]
        
        direction_filterwidth = 20
        forward = np.cross(across, np.array([[0,1,0]]))
        forward = filters.gaussian_filter1d(forward, direction_filterwidth, axis=0, mode='nearest')    
        forward = forward / np.sqrt((forward**2).sum(axis=-1))[...,np.newaxis]

        """ Remove Y Rotation """
        target = np.array([[0,0,1]]).repeat(len(forward), axis=0)
        rotation = Quaternions.between(forward, target)[:,np.newaxis]    
        positions = rotation * positions
        
        """ Get Root Rotation """
        velocity = rotation[1:] * velocity
        rvelocity = Pivots.from_quaternions(rotation[1:] * -rotation[:-1]).ps
        
        """ Add Velocity, RVelocity, Foot Contacts to vector """
        positions = positions[:-1]
        positions = positions.reshape(len(positions), -1)
        positions = np.concatenate([positions, velocity[:,:,0]], axis=-1)
        positions = np.concatenate([positions, velocity[:,:,2]], axis=-1)
        positions = np.concatenate([positions, rvelocity], axis=-1)
        positions = np.concatenate([positions, feet_l, feet_r], axis=-1)

        speechData_list.append(speechData) #Save speech info
        positions_list.append(positions)    #Save skeleton info

    if len(positions_list[0]) != len(positions_list[1]): raise Exception()
    if len(positions_list[1]) != len(positions_list[2]): raise Exception()

    if len(speechData_list[0]) != len(speechData_list[1]): raise Exception()
    if len(speechData_list[1]) != len(speechData_list[2]): raise Exception()
        
    """ Slide over windows """
    windows = [list(),list(),list()]
    windows_speech = [list(),list(),list()]
    
    print("skelSize {0} vs speechSize {1}".format(positions.shape[0],speechData['indicator'].shape[0]))
    for j in range(0, len(positions)-window//8, window_step):

        for pIdx in range(len(positions_list)):
        
            """ If slice too small pad out by repeating start and end poses """
            slice = positions_list[pIdx][j:j+window]
            if len(slice) < window:
                break
                # left  = slice[:1].repeat((window-len(slice))//2 + (window-len(slice))%2, axis=0)
                # left[:,-7:-4] = 0.0
                # right = slice[-1:].repeat((window-len(slice))//2, axis=0)
                # right[:,-7:-4] = 0.0
                # slice = np.concatenate([left, slice, right], axis=0)
            
            if len(slice) != window: raise Exception()
            
            windows[pIdx].append(slice)
            
            # """ Find Class """
            # cls = -1
            # if filename.startswith('hdm05'):
            #     cls_name = os.path.splitext(os.path.split(filename)[1])[0][7:-8]
            #     cls = class_names.index(class_map[cls_name]) if cls_name in class_map else -1
            # if filename.startswith('styletransfer'):
            #     cls_name = os.path.splitext(os.path.split(filename)[1])[0]
            #     cls = np.array([
            #         styletransfer_motions.index('_'.join(cls_name.split('_')[1:-1])),
            #         styletransfer_styles.index(cls_name.split('_')[0])])
            cls = speechData_list[pIdx]['indicator'][j:j+window]
            windows_speech[pIdx].append(cls)
        
    return windows, windows_speech



def get_files(directory):
    return [os.path.join(directory,f) for f in sorted(list(os.listdir(directory)))
    if os.path.isfile(os.path.join(directory,f))
    and f.endswith('.bvh') and f != 'rest.bvh'] 


testingSet = ['170221_haggling_b1','170221_haggling_b2','170221_haggling_b3','170228_haggling_b1','170228_haggling_b2','170228_haggling_b3']

def get_files_haggling_testing(directory):
    fileListInit = [os.path.join(directory,f) for f in sorted(list(os.listdir(directory)))
    if os.path.isfile(os.path.join(directory,f))
    and f.endswith('.bvh') and f != 'rest.bvh'] 

    fileList=list()
    for f in fileListInit:
        bTesting = False
        for keyword in testingSet :
            if keyword in f:
                bTesting = True
                break
        if bTesting:
            fileList.append(f)
    return fileList

def get_files_haggling_training(directory):
    fileListInit = [os.path.join(directory,f) for f in sorted(list(os.listdir(directory)))
    if os.path.isfile(os.path.join(directory,f))
    and f.endswith('.bvh') and f != 'rest.bvh'] 

    fileList=list()

    for f in fileListInit:
        bTesting = False
        for keyword in testingSet :
            if keyword in f:
                bTesting = True
                break
        if not bTesting:
            fileList.append(f)

    return fileList

#p0: buyer, p1: winner, p2: loser
def get_files_haggling_winners(directory):
    return [os.path.join(directory,f) for f in sorted(list(os.listdir(directory)))
    if os.path.isfile(os.path.join(directory,f))
    and f.endswith('.bvh') and f != 'rest.bvh' and '_p1' in f] 

def get_files_haggling_losers(directory):
    return [os.path.join(directory,f) for f in sorted(list(os.listdir(directory)))
    if os.path.isfile(os.path.join(directory,f))
    and f.endswith('.bvh') and f != 'rest.bvh' and '_p2' in f] 


#Dividing Testing/training 
def get_files_haggling_winners_div(directory, bReturnTesting):
    fileListInit = [os.path.join(directory,f) for f in sorted(list(os.listdir(directory)))
    if os.path.isfile(os.path.join(directory,f))
        and f.endswith('.bvh') and f != 'rest.bvh' and '_p1' in f]

    fileList=list()

    for f in fileListInit:
        bTesting = False
        for keyword in testingSet :
            if keyword in f:
                bTesting = True
                break

        if bReturnTesting and bTesting:
                fileList.append(f)
        if not bReturnTesting and not bTesting:
                fileList.append(f)

    return fileList


#Dividing Testing/training 
def get_files_haggling_losers_div(directory, bReturnTesting):
    fileListInit =  [os.path.join(directory,f) for f in sorted(list(os.listdir(directory)))
    if os.path.isfile(os.path.join(directory,f))
    and f.endswith('.bvh') and f != 'rest.bvh' and '_p2' in f] 

    fileList=list()

    for f in fileListInit:
        bTesting = False
        for keyword in testingSet :
            if keyword in f:
                bTesting = True
                break

        if bReturnTesting and bTesting:
                fileList.append(f)
        if not bReturnTesting and not bTesting:
                fileList.append(f)

    return fileList

""" White list with good body reconstruction quality"""
white_list_body = ['170221_haggling_b1_group0',
'170221_haggling_b1_group2',
'170221_haggling_b1_group3',
'170221_haggling_b1_group4',
'170221_haggling_b2_group1',
'170221_haggling_b2_group2',
'170221_haggling_b2_group4',
'170221_haggling_b2_group5',
'170221_haggling_b3_group0',
'170221_haggling_b3_group1',
'170221_haggling_b3_group2',
'170228_haggling_b1_group0',
'170228_haggling_b1_group1',
'170228_haggling_b1_group2',
'170228_haggling_b1_group3',
'170228_haggling_b1_group6',
'170228_haggling_b1_group7',
'170228_haggling_b1_group8',
'170228_haggling_b1_group9',
'170221_haggling_m1_group0',
'170221_haggling_m1_group2',
'170221_haggling_m1_group3',
'170221_haggling_m1_group4',
'170221_haggling_m1_group5',
'170221_haggling_m2_group2',
'170221_haggling_m2_group3',
'170221_haggling_m2_group5',
'170221_haggling_m3_group0',
'170221_haggling_m3_group1',
'170221_haggling_m3_group2',
'170224_haggling_a1_group0',
'170224_haggling_a1_group1',
'170224_haggling_a1_group3',
'170224_haggling_a1_group4',
'170224_haggling_a1_group5',
'170224_haggling_a1_group6',
'170224_haggling_a2_group0',
'170224_haggling_a2_group1',
'170224_haggling_a2_group2',
'170224_haggling_a2_group6',
'170224_haggling_a3_group0',
'170224_haggling_b1_group0',
'170224_haggling_b1_group4',
'170224_haggling_b1_group5',
'170224_haggling_b1_group6',
'170224_haggling_b2_group0',
'170224_haggling_b2_group1',
'170224_haggling_b2_group4',
'170224_haggling_b2_group5',
'170224_haggling_b2_group7',
'170224_haggling_b3_group0',
'170224_haggling_b3_group2',
'170228_haggling_a1_group0',
'170228_haggling_a1_group1',
'170228_haggling_a1_group4',
'170228_haggling_a1_group6',
'170228_haggling_a2_group0',
'170228_haggling_a2_group1',
'170228_haggling_a2_group2',
'170228_haggling_a2_group4',
'170228_haggling_a2_group5',
'170228_haggling_a2_group6',
'170228_haggling_a2_group7',
'170228_haggling_a3_group1',
'170228_haggling_a3_group2',
'170228_haggling_a3_group3',
'170228_haggling_b2_group0',
'170228_haggling_b2_group1',
'170228_haggling_b2_group4',
'170228_haggling_b2_group5',
'170228_haggling_b2_group8',
'170228_haggling_b3_group0',
'170228_haggling_b3_group1',
'170228_haggling_b3_group2',
'170228_haggling_b3_group3',
'170404_haggling_a1_group2',
'170404_haggling_a2_group1',
'170404_haggling_a2_group2',
'170404_haggling_a2_group3',
'170404_haggling_a3_group0',
'170404_haggling_a3_group1',
'170404_haggling_b1_group3',
'170404_haggling_b1_group6',
'170404_haggling_b1_group7',
'170404_haggling_b2_group1',
'170404_haggling_b2_group4',
'170404_haggling_b2_group6',
'170404_haggling_b3_group1',
'170404_haggling_b3_group2',
'170407_haggling_a1_group1',
'170407_haggling_a1_group3',
'170407_haggling_a1_group5',
'170407_haggling_a2_group3',
'170407_haggling_a2_group5',
'170407_haggling_b1_group0',
'170407_haggling_b1_group1',
'170407_haggling_b1_group2',
'170407_haggling_b1_group3',
'170407_haggling_b1_group4',
'170407_haggling_b1_group6',
'170407_haggling_b1_group7',
'170407_haggling_b2_group0',
'170407_haggling_b2_group1',
'170407_haggling_b2_group2',
'170407_haggling_b2_group4',
'170407_haggling_b2_group5',
'170407_haggling_b2_group6']


def get_files_haggling_buyers(directory, bReturnTesting):
    fileListInit =  [os.path.join(directory,f) for f in sorted(list(os.listdir(directory)))
    if os.path.isfile(os.path.join(directory,f))
    and f.endswith('.bvh') and f != 'rest.bvh' and '_p0' in f] 

    fileList=list()

    for f in fileListInit:
        bTesting = False

        bInWhiteList = False
        for keyword in white_list_body:
            if keyword in f:
                bInWhiteList = True
                break

        if not bInWhiteList:
            continue

        for keyword in testingSet :
            if keyword in f:
                bTesting = True
                break

        if bReturnTesting and bTesting:     #Testing
            fileList.append(f)
        if not bReturnTesting and not bTesting:  #Training
            fileList.append(f)

    return fileList

def get_files_haggling_sellers(directory, bReturnTesting):
    fileListInit = [os.path.join(directory,f) for f in sorted(list(os.listdir(directory)))
    if os.path.isfile(os.path.join(directory,f))
    and f.endswith('.bvh') and f != 'rest.bvh' and '_p0' not in f] 

    fileList=list()

    for f in fileListInit:
        bTesting = False

        bInWhiteList = False
        for keyword in white_list_body:
            if keyword in f:
                bInWhiteList = True
                break

        if not bInWhiteList:
            continue

        for keyword in testingSet :
            if keyword in f:
                bTesting = True
                break

        if bReturnTesting and bTesting:     #Testing
            fileList.append(f)
        if not bReturnTesting and not bTesting:  #Training
            fileList.append(f)

    return fileList


def get_files_haggling(directory, bReturnTesting):
    fileListInit = [os.path.join(directory,f) for f in sorted(list(os.listdir(directory)))
    if os.path.isfile(os.path.join(directory,f))
    and f.endswith('.pkl') and f != 'rest.bvh'] 

    fileList=list()

    for f in fileListInit:
        bTesting = False

        bInWhiteList = False
        #for keyword in white_list_body:
        for keyword in white_list_body:
            if keyword in f:
                bInWhiteList = True
                break

        if not bInWhiteList:
            continue

        for keyword in testingSet :
            if keyword in f:
                bTesting = True
                break

        if bReturnTesting and bTesting:     #Testing
            fileList.append(f)
        if not bReturnTesting and not bTesting:  #Training
            fileList.append(f)

    return fileList



"""
    output:
        - return windows_pos_normal:  [elm1,elm2,elm3], where elmX is list of (window x 9) and the depth order is pos(3), faceNormal(3), bodyNormal(3)
        - windows_speech: [elm1,elm2,elm3], where elmX is list of (window,)
"""
import cPickle as pickle
def process_file_withSpeech_byGroup_normalized(filename, window=240, window_step=120, apply_brl=False):
    
    #anim, names, frametime = BVH.load(filename)
    motionData = pickle.load( open(filename, "rb" ) )

    #Load speech info
    seqName = os.path.basename(filename)
    #speech_fileName = seqName[:-7] + '.pkl'
    speech_fileName = seqName
    speechPath = './panopticDB_pkl_speech_hagglingProcessed/' +speech_fileName
    speechData_raw = pickle.load( open( speechPath, "rb" ) )

    motionData_list=list()
    speechData_list=list()
    
    for pIdx in range(0,3):
        
        speechData = speechData_raw['speechData'][pIdx]
        speechData_list.append(speechData) #Save speech info

        b_normal = motionData['subjects'][pIdx]['bodyNormal']       #(3,frames)
        f_normal = motionData['subjects'][pIdx]['faceNormal']       #(3,frames)
        pos = motionData['subjects'][pIdx]['joints19'][:3,:]        #(3,frames)

        b_normal = np.swapaxes(b_normal,0,1)       #(frames, 3)
        f_normal = np.swapaxes(f_normal,0,1)       #(frames, 3)
        pos = np.swapaxes(pos,0,1)            #(frames, 3)

        motionData_list.append({'faceNormal': f_normal, 'bodyNormal': b_normal, 'pos':pos})    #all data (frames x 3)

    #Normalize
    refPos = motionData_list[0]['pos']  #(frames, 3)
    refFaceNormal = motionData_list[0]['faceNormal'] #(frames, 3)
    refBodyNormal = motionData_list[0]['bodyNormal'] #(frames, 3)

    target = np.array([[0,0,1]]).repeat(len(refBodyNormal), axis=0)
    #target = np.array([[0,0,-1]]).repeat(len(refBodyNormal), axis=0)
    refRotation = Quaternions.between(refBodyNormal, target)[:,np.newaxis]       #(frames, 1)

    # """Debug"""
    # bodyNormal_scalar  = Pivots.from_directions(refBodyNormal).ps
    # import matplotlib.pyplot as plt
    # ax2=plt.subplot(111)
    # plt.plot(bodyNormal_scalar)
    # plt.title('bodyNormal_scalar')
    # plt.show()

    for pIdx in range(0,3):

        """ refPos on the Origin """
        pos =  (motionData_list[pIdx]['pos'] - refPos)
        pos = np.expand_dims(pos,1)
        motionData_list[pIdx]['pos'] = refRotation * pos
        motionData_list[pIdx]['pos'] = motionData_list[pIdx]['pos'][:,0,:]
        

        """ Orientation as a vector """
        bodyNormal_scalar = refRotation*  np.expand_dims(motionData_list[pIdx]['bodyNormal'],1)
        motionData_list[pIdx]['bodyNormal'] = bodyNormal_scalar[:,0,:]

        faceNormal_scalar = refRotation * np.expand_dims(motionData_list[pIdx]['faceNormal'],1)
        motionData_list[pIdx]['faceNormal'] = faceNormal_scalar[:,0,:]


        """ Orientation as a single scalar """
        """
        bodyNormal_scalar = rotation*  np.expand_dims(motionData_list[pIdx]['bodyNormal'],1)
        bodyNormal_scalar  = Pivots.from_directions(bodyNormal_scalar).ps
        bodyNormal_scalar = np.expand_dims(bodyNormal_scalar,1)
        motionData_list[pIdx]['bodyNormal'] = bodyNormal_scalar[:,0,:]

        faceNormal_scalar = rotation * np.expand_dims(motionData_list[pIdx]['faceNormal'],1)
        faceNormal_scalar  = Pivots.from_directions(faceNormal_scalar).ps
        faceNormal_scalar = np.expand_dims(faceNormal_scalar,1)
        motionData_list[pIdx]['faceNormal'] = faceNormal_scalar[:,0,:]
        """

        # """Debug"""
        # import matplotlib.pyplot as plt
        # ax2=plt.subplot(211)
        # plt.plot(faceNormal_scalar.flatten())
        # plt.title('faceNormal_scalar')
        # ax2=plt.subplot(212)
        # plt.plot(bodyNormal_scalar.flatten())
        # plt.title('bodyNormal_scalar')
        # plt.show()


    if len(motionData_list[0]['pos']) != len(motionData_list[1]['pos']): raise Exception()
    if len(motionData_list[1]['pos']) != len(motionData_list[2]['pos']): raise Exception()
    if len(speechData_list[0]) != len(speechData_list[1]): raise Exception()
    if len(speechData_list[1]) != len(speechData_list[2]): raise Exception()
        
    """ Slide over windows """
    windows_pos_normal = [list(),list(),list()]
    windows_speech = [list(),list(),list()]
    
    frameNum = motionData_list[0]['pos'].shape[0]

    
    #Original ordering: 0:buyer 1:winner 2:loser
    pIdx_brl = [0, 1,2]     

    if apply_brl:       #New ordering: 0:buyer 1:right 2:left  (w.r.t buyer)
        if motionData['rightSellerId'] != motionData['winnerId']:
            pIdx_brl = [0, 2, 1]     

    print("skelSize {0} vs speechSize {1}".format(frameNum,speechData['indicator'].shape[0]))
    for j in range(0, frameNum - window, window_step):

        #for pIdx in range(len(motionData_list)):
        for outputIdx, pIdx in enumerate(pIdx_brl):
        

            slice = motionData_list[pIdx]['pos'][j:j+window] #frames x 3
            if len(slice) < window:
                break
            if len(slice) != window: raise Exception()
            slice_concat = slice

            slice = motionData_list[pIdx]['faceNormal'][j:j+window] #frames x 3
            if len(slice) < window:
                break
            if len(slice) != window: raise Exception()
            slice_concat = np.concatenate((slice_concat,slice), axis =1) #frame x6

            slice = motionData_list[pIdx]['bodyNormal'][j:j+window] #frames x 3
            if len(slice) < window:
                break
            if len(slice) != window: raise Exception()
            slice_concat = np.concatenate((slice_concat,slice), axis =1) #frame x6

            windows_pos_normal[outputIdx].append(slice_concat)

            #Speech Data
            speechSignal = speechData_list[pIdx]['indicator'][j:j+window]
            windows_speech[outputIdx].append(speechSignal)

    return windows_pos_normal, windows_speech


def process_file_withSpeech_byGroup_normalized_byFirstFrame(filename, window=240, window_step=120, apply_brl=False):
    
    #anim, names, frametime = BVH.load(filename)
    motionData = pickle.load( open(filename, "rb" ) )

    #Load speech info
    seqName = os.path.basename(filename)
    #speech_fileName = seqName[:-7] + '.pkl'
    speech_fileName = seqName
    speechPath = './panopticDB_pkl_speech_hagglingProcessed/' +speech_fileName
    speechData_raw = pickle.load( open( speechPath, "rb" ) )

    motionData_list=list()
    speechData_list=list()
    
    for pIdx in range(0,3):
        
        speechData = speechData_raw['speechData'][pIdx]
        speechData_list.append(speechData) #Save speech info

        b_normal = motionData['subjects'][pIdx]['bodyNormal']       #(3,frames)
        f_normal = motionData['subjects'][pIdx]['faceNormal']       #(3,frames)
        pos = motionData['subjects'][pIdx]['joints19'][:3,:]        #(3,frames)

        b_normal = np.swapaxes(b_normal,0,1)       #(frames, 3)
        f_normal = np.swapaxes(f_normal,0,1)       #(frames, 3)
        pos = np.swapaxes(pos,0,1)            #(frames, 3)

        motionData_list.append({'faceNormal': f_normal, 'bodyNormal': b_normal, 'pos':pos})    #all data (frames x 3)

    #Normalize
    refPos = motionData_list[0]['pos']  #(frames, 3)
    refFaceNormal = motionData_list[0]['faceNormal'] #(frames, 3)
    refBodyNormal = motionData_list[0]['bodyNormal'] #(frames, 3)

    target = np.array([[0,0,1]]).repeat(len(refBodyNormal), axis=0)
    #target = np.array([[0,0,-1]]).repeat(len(refBodyNormal), axis=0)
    refRotation = Quaternions.between(refBodyNormal, target)[:,np.newaxis]       #(frames, 1)

    # """Debug"""
    # bodyNormal_scalar  = Pivots.from_directions(refBodyNormal).ps
    # import matplotlib.pyplot as plt
    # ax2=plt.subplot(111)
    # plt.plot(bodyNormal_scalar)
    # plt.title('bodyNormal_scalar')
    # plt.show()

    # for pIdx in range(0,3):

    #     """ refPos on the Origin """
    #     pos =  (motionData_list[pIdx]['pos'] - refPos[0])
    #     motionData_list[pIdx]['pos'] = pos
    #     #pos = np.expand_dims(pos,1)
    #     #motionData_list[pIdx]['pos'] = refRotation[0:1] * pos
    #     #motionData_list[pIdx]['pos'] = motionData_list[pIdx]['pos'][:,0,:]
        

    #     """ Orientation as a vector """
    #     bodyNormal_scalar = refRotation[0:1]*  np.expand_dims(motionData_list[pIdx]['bodyNormal'],1)
    #     motionData_list[pIdx]['bodyNormal'] = bodyNormal_scalar[:,0,:]

    #     faceNormal_scalar = refRotation[0:1] * np.expand_dims(motionData_list[pIdx]['faceNormal'],1)
    #     motionData_list[pIdx]['faceNormal'] = faceNormal_scalar[:,0,:]




    if len(motionData_list[0]['pos']) != len(motionData_list[1]['pos']): raise Exception()
    if len(motionData_list[1]['pos']) != len(motionData_list[2]['pos']): raise Exception()
    if len(speechData_list[0]) != len(speechData_list[1]): raise Exception()
    if len(speechData_list[1]) != len(speechData_list[2]): raise Exception()
        
    """ Slide over windows """
    windows_pos_normal = [list(),list(),list()]
    windows_speech = [list(),list(),list()]
    
    frameNum = motionData_list[0]['pos'].shape[0]

    
    #Original ordering: 0:buyer 1:winner 2:loser
    pIdx_brl = [0, 1,2]     

    if apply_brl:       #New ordering: 0:buyer 1:right 2:left  (w.r.t buyer)
        if motionData['rightSellerId'] != motionData['winnerId']:
            pIdx_brl = [0, 2, 1]     

    print("skelSize {0} vs speechSize {1}".format(frameNum,speechData['indicator'].shape[0]))
    for j in range(0, frameNum - window, window_step):

        #for pIdx in range(len(motionData_list)):
        for outputIdx, pIdx in enumerate(pIdx_brl):


            refPosSlice =  refPos[j]
            refRotSlice = refRotation[j:j+1]

            #slice = motionData_list[pIdx]['pos'][j:j+window] #frames x 3
            slice = motionData_list[pIdx]['pos'][j:j+window] - refPosSlice#frames x 3
            if len(slice) < window:
                break
            if len(slice) != window: raise Exception()
            slice_concat = slice

            slice = motionData_list[pIdx]['faceNormal'][j:j+window] #frames x 3
            if len(slice) < window:
                break
            if len(slice) != window: raise Exception()
            slice_concat = np.concatenate((slice_concat,slice), axis =1) #frame x6

            slice = motionData_list[pIdx]['bodyNormal'][j:j+window] #frames x 3
            if len(slice) < window:
                break
            if len(slice) != window: raise Exception()
            slice_concat = np.concatenate((slice_concat,slice), axis =1) #frame x6

            windows_pos_normal[outputIdx].append(slice_concat)

            #Speech Data
            speechSignal = speechData_list[pIdx]['indicator'][j:j+window]
            windows_speech[outputIdx].append(speechSignal)

    return windows_pos_normal, windows_speech



"""
    output:
        - return windows_pos_normal:  [elm1,elm2,elm3], where elmX is list of (window x 9) and the depth order is pos(3), faceNormal(3), bodyNormal(3)
        - windows_speech: [elm1,elm2,elm3], where elmX is list of (window,)
"""
import cPickle as pickle
def process_file_withSpeech_byGroup(filename, window=240, window_step=120, apply_brl=False):
    
    #anim, names, frametime = BVH.load(filename)
    motionData = pickle.load( open(filename, "rb" ) )

    #Load speech info
    seqName = os.path.basename(filename)
    #speech_fileName = seqName[:-7] + '.pkl'
    speech_fileName = seqName
    speechPath = './panopticDB_pkl_speech_hagglingProcessed/' +speech_fileName
    speechData_raw = pickle.load( open( speechPath, "rb" ) )

    motionData_list=list()
    speechData_list=list()
    
    for pIdx in range(0,3):
        
        speechData = speechData_raw['speechData'][pIdx]
        speechData_list.append(speechData) #Save speech info

        b_normal = motionData['subjects'][pIdx]['bodyNormal']
        f_normal = motionData['subjects'][pIdx]['faceNormal']
        pos = motionData['subjects'][pIdx]['joints19'][:3,:]

        motionData_list.append({'faceNormal': np.swapaxes(f_normal,0,1), 'bodyNormal': np.swapaxes(b_normal,0,1), 'pos':np.swapaxes(pos,0,1)})    #all data (frames x 3)

    """compute face normal as an angle"""
    for pIdx in range(0,3):

        #Assuming, 0,1,2 are in BuyerRightLeft order. me->rightPerson direction is labelled as 0. 
        if pIdx==0: #buyer
            pos2Other_0 = motionData_list[1]['pos'] - motionData_list[pIdx]['pos']  #(frames,3)
            pos2Other_1 = motionData_list[2]['pos'] - motionData_list[pIdx]['pos']  #(frames,3)
        elif pIdx ==1:  #right person w.r.t buyer
            pos2Other_0 = motionData_list[2]['pos'] - motionData_list[pIdx]['pos']
            pos2Other_1 = motionData_list[0]['pos'] - motionData_list[pIdx]['pos']
        elif pIdx ==2:  #left person
            pos2Other_0 = motionData_list[0]['pos'] - motionData_list[pIdx]['pos']
            pos2Other_1 = motionData_list[1]['pos'] - motionData_list[pIdx]['pos']
        
        #
        pos2Other_0 = normalize(pos2Other_0,axis=1) #(frames,3)
        pos2Other_1 = normalize(pos2Other_1,axis=1) #(frames,3)
        faceNormal = motionData_list[pIdx]['faceNormal']
        bodyNormal = motionData_list[pIdx]['bodyNormal']

        rotation = Quaternions.between(pos2Other_0, pos2Other_1)[:,np.newaxis]     #rot from 0->1
        angleValue_full = Pivots.from_quaternions(rotation).ps

        rotation = Quaternions.between(pos2Other_0, faceNormal)[:,np.newaxis]     #rot from 0->1
        angleValue_faceDir = Pivots.from_quaternions(rotation).ps
        attentionLabel_face = angleValue_faceDir/ angleValue_full    #0 for attention on the right person, 1 for on left person

        rotation = Quaternions.between(pos2Other_0, bodyNormal)[:,np.newaxis]     #rot from 0->1
        angleValue_bodyDir = Pivots.from_quaternions(rotation).ps
        attentionLabel_body = angleValue_bodyDir / angleValue_full    #0 for attention on the right person, 1 for on left person

        motionData_list[pIdx]['binaryFaceAtt'] = attentionLabel_face
        motionData_list[pIdx]['binaryBodyAtt'] = attentionLabel_body



    if len(motionData_list[0]['pos']) != len(motionData_list[1]['pos']): raise Exception()
    if len(motionData_list[1]['pos']) != len(motionData_list[2]['pos']): raise Exception()
    if len(speechData_list[0]) != len(speechData_list[1]): raise Exception()
    if len(speechData_list[1]) != len(speechData_list[2]): raise Exception()
        
    """ Slide over windows """
    windows_pos_normal = [list(),list(),list()]
    windows_speech = [list(),list(),list()]
    windows_attention =[list(),list(),list()]

    #Original ordering: 0:buyer 1:winner 2:loser
    pIdx_brl = [0, 1,2]     
    bSwtchedBRL = False
    if apply_brl:       #New ordering: 0:buyer 1:right 2:left  (w.r.t buyer)
        if motionData['rightSellerId'] != motionData['winnerId']:
            pIdx_brl = [0, 2, 1]     
            bSwtchedBRL = True

    frameNum = motionData_list[0]['pos'].shape[0]
    print("skelSize {0} vs speechSize {1}".format(frameNum,speechData['indicator'].shape[0]))
    for j in range(0, frameNum - window, window_step):

        bGood = True
        for outputIdx, pIdx in enumerate(pIdx_brl):
            #Binary attention data
            atten = motionData_list[pIdx]['binaryFaceAtt'][j:j+window]  #(frame,1)
            slice = motionData_list[pIdx]['binaryBodyAtt'][j:j+window]  #(frame,1)
            atten_concat = np.concatenate((atten,slice), axis =1) #frame x6

            nanCheck = np.max(atten_concat)
            if nanCheck!=nanCheck:
                print('nan is detected')
                bGood = False
                break


        if bGood ==False:
            continue
        
        for outputIdx, pIdx in enumerate(pIdx_brl):
        #for pIdx in range(len(motionData_list)):
        
            slice = motionData_list[pIdx]['pos'][j:j+window] #frames x 3
            if len(slice) < window:
                break
            if len(slice) != window: raise Exception()
            slice_concat = slice

            slice = motionData_list[pIdx]['faceNormal'][j:j+window] #frames x 3
            if len(slice) < window:
                break
            if len(slice) != window: raise Exception()
            slice_concat = np.concatenate((slice_concat,slice), axis =1) #frame x6

            slice = motionData_list[pIdx]['bodyNormal'][j:j+window] #frames x 3
            if len(slice) < window:
                break
            if len(slice) != window: raise Exception()
            slice_concat = np.concatenate((slice_concat,slice), axis =1) #frame x6

            #Binary attention data
            atten = motionData_list[pIdx]['binaryFaceAtt'][j:j+window]  #(frame,1)
            slice = motionData_list[pIdx]['binaryBodyAtt'][j:j+window]  #(frame,1)
            atten_concat = np.concatenate((atten,slice), axis =1) #frame x6

            if bSwtchedBRL:
                #ptIdx 1 and 2 are swithced, #Flip 0,1 label for all ptIdx
                atten_concat = 1 - atten_concat
        
            windows_pos_normal[outputIdx].append(slice_concat)

            #Speech Data
            speechSignal = speechData_list[pIdx]['indicator'][j:j+window]
            windows_speech[outputIdx].append(speechSignal)


            windows_attention[outputIdx].append(atten_concat)
    
        

    return windows_pos_normal, windows_speech, windows_attention


"""Haggling testing games sellers"""
faceParamDir = '/ssd/codes/pytorch_motionSynth/motionsynth_data/data/processed_panoptic/panopticDB_pkl_hagglingProcessed'
h36m_files = get_files_haggling(faceParamDir,True)
print('Num: {}'.format(len(h36m_files)))
group_data = [ [], [], [] ]
group_speech = [ [], [], [] ]
group_attention = [ [], [], [] ]
for i, item in enumerate(h36m_files):

    # if not ('170407_haggling_b2_group6' in item):
    #     continue
    print('Processing %i of %i (%s)' % (i, len(h36m_files), item))
    #clips, speech = process_file_withSpeech_byGroup_normalized(item,240,5, apply_brl=True)
    clips, speech, attention = process_file_withSpeech_byGroup(item,240,5, apply_brl=True)
    #clips, speech = process_file_withSpeech_byGroup_normalized_byFirstFrame(item,240,5, apply_brl=True)

    for pIdx in range(3):
        group_data[pIdx] += clips[pIdx]
        group_speech[pIdx] += speech[pIdx]
        group_attention[pIdx] +=  attention[pIdx]
        #np.concatenate( (group_attention[pIdx],attention[pIdx]),axis=0)

    if i==5:
        break
for pIdx in range(3):
    group_data[pIdx] = np.array(group_data[pIdx])       #(num, frames, featureNum:9)
    group_speech[pIdx] = np.array(group_speech[pIdx])   #(num, frames, featureNum:9)
    group_attention[pIdx] = np.array(group_attention[pIdx])   #(num, frames, featureNum:2)

print("Data shape: {}".format(group_data[0].shape))
np.savez('data_hagglingSellers_speech_formation_240frm_5gap_white_brl_atten_testing_tiny', clips=group_data, speech=group_speech, attention=group_attention)
#np.savez('data_hagglingSellers_speech_formation_240frm_5gap_white_brl_firstFrmPosNorm_testing', clips=group_data, speech=group_speech)
#np.savez('data_hagglingSellers_speech_formation_pN_rN_rV_240frm_5gap_white_brl_testing', clips=group_data, speech=group_speech)