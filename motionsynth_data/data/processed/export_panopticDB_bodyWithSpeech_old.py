"""
Needs speech file: panopticDB_pkl_speech_hagglingProcessed
This file is from: /ssd/codes/haggling_audio

"""


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




import pickle
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

def get_files_haggling_buyers(directory):
    return [os.path.join(directory,f) for f in sorted(list(os.listdir(directory)))
    if os.path.isfile(os.path.join(directory,f))
    and f.endswith('.bvh') and f != 'rest.bvh' and '_p0' in f] 

def get_files_haggling_sellers(directory, bReturnTesting):
    fileListInit = [os.path.join(directory,f) for f in sorted(list(os.listdir(directory)))
    if os.path.isfile(os.path.join(directory,f))
    and f.endswith('.bvh') and f != 'rest.bvh' and '_p0' not in f] 

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


"""Haggling training games sellers"""
h36m_files = get_files_haggling_sellers('panoptic',False)
print(h36m_files)
h36m_clips = []
h36m_classes = []
for i, item in enumerate(h36m_files):
    print('Processing %i of %i (%s)' % (i, len(h36m_files), item))
    clips, speech = process_file_withSpeech(item,1,1)
    h36m_clips += clips
    h36m_classes += speech   
data_clips = np.array(h36m_clips)
data_speech = np.array(h36m_classes)
np.savez_compressed('data_panoptic_speech_haggling_sellers_training_byFrame', clips=data_clips, classes=data_speech)


"""Haggling training games sellers"""
h36m_files = get_files_haggling_sellers('panoptic',True)
print(h36m_files)
h36m_clips = []
h36m_classes = []
for i, item in enumerate(h36m_files):
    print('Processing %i of %i (%s)' % (i, len(h36m_files), item))
    clips, speech = process_file_withSpeech(item,1,1)
    h36m_clips += clips
    h36m_classes += speech   
data_clips = np.array(h36m_clips)
data_speech = np.array(h36m_classes)
np.savez_compressed('data_panoptic_speech_haggling_sellers_testing_byFrame', clips=data_clips, classes=data_speech)


# """Haggling training games sellers"""
# h36m_files = get_files_haggling_sellers('panoptic',False)
# print(h36m_files)
# h36m_clips = []
# h36m_speech = []
# for i, item in enumerate(h36m_files):
#     print('Processing %i of %i (%s)' % (i, len(h36m_files), item))
#     clips, cls = process_file(item)
#     h36m_clips += clips
#     h36m_classes += cls    
# data_clips = np.array(h36m_clips)
# data_speech = np.array(h36m_speech)
# np.savez_compressed('data_panoptic_speech_haggling_sellers_testing', clips=data_clips, speech=data_speech)


# """Haggling training games winners"""
# h36m_files = get_files_haggling_winners_div('panoptic',False)
# print(h36m_files)
# h36m_clips = []
# h36m_classes = []
# for i, item in enumerate(h36m_files):
#     print('Processing %i of %i (%s)' % (i, len(h36m_files), item))
#     clips, cls = process_file(item)
#     h36m_clips += clips
#     h36m_classes += cls    
# data_clips = np.array(h36m_clips)
# data_classes = np.array(h36m_classes)
# np.savez_compressed('data_panoptic_haggling_winners_training', clips=data_clips, classes=data_classes)


# """Haggling training games losers"""
# h36m_files = get_files_haggling_losers_div('panoptic',True)
# print(h36m_files)
# h36m_clips = []
# h36m_classes = []
# for i, item in enumerate(h36m_files):
#     print('Processing %i of %i (%s)' % (i, len(h36m_files), item))
#     clips, cls = process_file(item)
#     h36m_clips += clips
#     h36m_classes += cls    
# data_clips = np.array(h36m_clips)
# data_classes = np.array(h36m_classes)
# np.savez_compressed('data_panoptic_haggling_losers_testing', clips=data_clips, classes=data_classes)



# """Haggling training games losers"""
# h36m_files = get_files_haggling_losers_div('panoptic',False)
# print(h36m_files)
# h36m_clips = []
# h36m_classes = []
# for i, item in enumerate(h36m_files):
#     print('Processing %i of %i (%s)' % (i, len(h36m_files), item))
#     clips, cls = process_file(item)
#     h36m_clips += clips
#     h36m_classes += cls    
# data_clips = np.array(h36m_clips)
# data_classes = np.array(h36m_classes)
# np.savez_compressed('data_panoptic_haggling_losers_training', clips=data_clips, classes=data_classes)
