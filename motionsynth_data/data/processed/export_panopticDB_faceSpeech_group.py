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


white_list_face = ['170221_haggling_b1_group0',
'170221_haggling_b1_group1',
'170221_haggling_b1_group2',
'170221_haggling_b1_group3',
'170221_haggling_b1_group4',
'170221_haggling_b2_group0',
'170221_haggling_b2_group1',
'170221_haggling_b2_group4',
'170221_haggling_b3_group2',
'170221_haggling_m1_group2',
'170221_haggling_m1_group4',
'170221_haggling_m2_group0',
'170221_haggling_m2_group4',
'170224_haggling_a1_group1',
'170224_haggling_a1_group2',
'170224_haggling_a1_group3',
'170224_haggling_a1_group4',
'170224_haggling_a1_group5',
'170224_haggling_a2_group1',
'170224_haggling_a2_group3',
'170224_haggling_a2_group5',
'170224_haggling_a3_group0',
'170224_haggling_a3_group2',
'170224_haggling_a3_group3',
'170224_haggling_b1_group0',
'170224_haggling_b1_group3',
'170224_haggling_b1_group4',
'170224_haggling_b1_group5',
'170224_haggling_b1_group6',
'170224_haggling_b1_group7',
'170224_haggling_b2_group0',
'170224_haggling_b2_group1',
'170224_haggling_b2_group2',
'170224_haggling_b2_group3',
'170224_haggling_b2_group4',
'170224_haggling_b2_group5',
'170224_haggling_b2_group6',
'170224_haggling_b2_group7',
'170224_haggling_b3_group0',
'170224_haggling_b3_group1',
'170224_haggling_b3_group2',
'170228_haggling_a1_group0',
'170228_haggling_a1_group1',
'170228_haggling_a1_group3',
'170228_haggling_a1_group4',
'170228_haggling_a1_group5',
'170228_haggling_a1_group6',
'170228_haggling_a1_group7',
'170228_haggling_a2_group0',
'170228_haggling_a2_group1',
'170228_haggling_a2_group5',
'170228_haggling_a2_group6',
'170228_haggling_a2_group7',
'170228_haggling_a3_group0',
'170228_haggling_a3_group1',
'170228_haggling_a3_group3',
'170228_haggling_b1_group0',
'170228_haggling_b1_group2',
'170228_haggling_b1_group3',
'170228_haggling_b1_group4',
'170228_haggling_b1_group6',
'170228_haggling_b1_group7',
'170228_haggling_b1_group8',
'170228_haggling_b1_group9',
'170228_haggling_b2_group1',
'170228_haggling_b2_group2',
'170228_haggling_b2_group5',
'170228_haggling_b2_group7',
'170228_haggling_b2_group8',
'170228_haggling_b2_group9',
'170228_haggling_b3_group0',
'170228_haggling_b3_group1',
'170228_haggling_b3_group3',
'170404_haggling_a1_group2',
'170404_haggling_a1_group3',
'170404_haggling_a2_group0',
'170404_haggling_a2_group2',
'170404_haggling_a2_group3',
'170404_haggling_a3_group0',
'170404_haggling_a3_group1',
'170404_haggling_b1_group2',
'170404_haggling_b1_group3',
'170404_haggling_b1_group4',
'170404_haggling_b1_group5',
'170404_haggling_b1_group6',
'170404_haggling_b1_group7',
'170404_haggling_b2_group1',
'170404_haggling_b2_group3',
'170404_haggling_b2_group4',
'170404_haggling_b2_group6',
'170404_haggling_b2_group7',
'170404_haggling_b3_group0',
'170404_haggling_b3_group1',
'170404_haggling_b3_group2',
'170407_haggling_a1_group0',
'170407_haggling_a1_group1',
'170407_haggling_a1_group2',
'170407_haggling_a1_group3',
'170407_haggling_a1_group4',
'170407_haggling_a1_group5',
'170407_haggling_a2_group0',
'170407_haggling_a2_group1',
'170407_haggling_a2_group2',
'170407_haggling_a2_group4',
'170407_haggling_a2_group5',
'170407_haggling_a3_group0',
'170407_haggling_a3_group1',
'170407_haggling_a3_group2',
'170407_haggling_b1_group0',
'170407_haggling_b1_group1',
'170407_haggling_b1_group2',
'170407_haggling_b1_group3',
'170407_haggling_b1_group4',
'170407_haggling_b1_group5',
'170407_haggling_b1_group6',
'170407_haggling_b1_group7',
'170407_haggling_b2_group0',
'170407_haggling_b2_group1',
'170407_haggling_b2_group4',
'170407_haggling_b2_group5',
'170407_haggling_b2_group6',
'170407_haggling_b2_group7',
'170407_haggling_b3_group1',
'170407_haggling_b3_group2',
#0 fails
'170221_haggling_b2_group2',
'170221_haggling_b3_group1',
'170221_haggling_m3_group0',
'170224_haggling_a1_group6',
'170224_haggling_a2_group0',
'170224_haggling_a2_group2',
'170224_haggling_b1_group2',
'170228_haggling_a2_group2',
'170228_haggling_b1_group5',
'170228_haggling_b3_group2']


def process_file_withSpeech(filename, subjectRole,  window=240, window_step=120):
    
    #anim, names, frametime = BVH.load(filename)
    faceParamData = pickle.load( open(filename, "rb" ) )

    #Load speech info
    seqName = os.path.basename(filename)
    #speech_fileName = seqName[:-7] + '.pkl'
    speech_fileName = seqName
    speechPath = './panopticDB_pkl_speech_hagglingProcessed/' +speech_fileName
    speechData = pickle.load( open( speechPath, "rb" ) )

    subjectRole = int(subjectRole)


    faceParamData = faceParamData['subjects'][subjectRole]['face_exp']  #200 x frames
    faceParamData = np.swapaxes(faceParamData,0,1) # frames x 200
    speechData = speechData['speechData'][subjectRole]  #(frames,)
    
    # """ Convert to 60 fps """
    # anim = anim[::2]
    # Origianl Human3.6 has 50 fps. So, no smapling is done
    
    """ Slide over windows """
    windows = []
    windows_classes = []
    

    print("skelSize {0} vs speechSize {1}".format(faceParamData.shape[0], speechData['indicator'].shape[0]))
    for j in range(0, faceParamData.shape[0]-window//8, window_step):
    
        """ If slice too small pad out by repeating start and end poses """
        slice = faceParamData[j:j+window]
        if len(slice) < window:
            break
            # left  = slice[:1].repeat((window-len(slice))//2 + (window-len(slice))%2, axis=0)
            # left[:,-7:-4] = 0.0
            # right = slice[-1:].repeat((window-len(slice))//2, axis=0)
            # right[:,-7:-4] = 0.0
            # slice = np.concatenate([left, slice, right], axis=0)
        
        if len(slice) != window: raise Exception()
        
        windows.append(slice)
        
        cls = speechData['indicator'][j:j+window]
        windows_classes.append(cls)
        
    return windows, windows_classes

"""
    output:
        - return windows_pos_normal:  [elm1,elm2,elm3], where elmX is list of (window x 9) and the depth order is pos(3), faceNormal(3), bodyNormal(3)
        - windows_speech: [elm1,elm2,elm3], where elmX is list of (window,)
"""
def process_file_withSpeech_byGroup(filename, window=240, window_step=120, featureNum=200):
    
   
    #Load speech info
    seqName = os.path.basename(filename)
    #speech_fileName = seqName[:-7] + '.pkl'
    speech_fileName = seqName
    speechPath = './panopticDB_pkl_speech_hagglingProcessed/' +speech_fileName
    speechData_raw = pickle.load( open( speechPath, "rb" ) )

    motionData_list=list()
    speechData_list=list()

    faceParamData = pickle.load( open(filename, "rb" ) )
    
    for pIdx in range(0,3):
        
        speechData = speechData_raw['speechData'][pIdx]
        speechData_list.append(speechData) #Save speech info

        #anim, names, frametime = BVH.load(filename)
        faceParam = faceParamData['subjects'][pIdx]['face_exp']  #200 x frames
        faceParam = np.swapaxes(faceParam,0,1) # frames x 200

        motionData_list.append(faceParam)    #all data (frames x 3)


    if len(motionData_list[0]) != len(motionData_list[1]): raise Exception()
    if len(motionData_list[1]) != len(motionData_list[2]): raise Exception()
    if len(speechData_list[0]) != len(speechData_list[1]): raise Exception()
    if len(speechData_list[1]) != len(speechData_list[2]): raise Exception()
        
    """ Slide over windows """
    windows_data = [list(),list(),list()]
    windows_speech = [list(),list(),list()]
    
    frameNum = motionData_list[0].shape[0]
    print("skelSize {0} vs speechSize {1}".format(frameNum,speechData['indicator'].shape[0]))
    for j in range(0, frameNum - window, window_step):

        bSkip =False
        for pIdx in range(len(motionData_list)):
        
            slice = motionData_list[pIdx][j:j+window] #(frames, featureDim:200)
            if len(slice) < window:
                break
            if len(slice) != window: raise Exception()

            #Check Bad data (abs(value)>2)
            maxValue = np.max(abs(slice))
            if(maxValue>2.0):
                bSkip = True
                break
        if bSkip:
            continue

        for pIdx in range(len(motionData_list)):
        
            slice = motionData_list[pIdx][j:j+window] #(frames, featureDim:200)

            slice_concat = slice    #(windowLeng, 200)

            #only save part of them
            slice_concat = slice_concat[:,:featureNum]

            windows_data[pIdx].append(slice_concat)

            #Speech Data
            speechSignal = speechData_list[pIdx]['indicator'][j:j+window]
            windows_speech[pIdx].append(speechSignal)

    return windows_data, windows_speech

def get_files_haggling(directory, bReturnTesting):
    fileListInit = [os.path.join(directory,f) for f in sorted(list(os.listdir(directory)))
    if os.path.isfile(os.path.join(directory,f))
    and f.endswith('.pkl') and f != 'rest.bvh'] 

    fileList=list()

    for f in fileListInit:
        bTesting = False

        bInWhiteList = False
        #for keyword in white_list_body:
        for keyword in white_list_face:
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


# faceParamDir = '/ssd/codes/pytorch_motionSynth/motionsynth_data/data/processed_panoptic/panopticDB_faceMesh_pkl_hagglingProcessed'
# """Haggling training games sellers"""
# bTesting = False
# h36m_files = get_files_haggling(faceParamDir,bTesting)
# print('Num: {}'.format(len(h36m_files)))
# group_data = [ [], [], [] ]
# group_speech = [ [], [], [] ]

# cnt =0
# for i, item in enumerate(h36m_files):
#     print('Processing %i of %i (%s)' % (i, len(h36m_files), item))
#     clips, speech = process_file_withSpeech_byGroup(item, 30, 10)

#     for pIdx in range(3):
#         group_data[pIdx] += clips[pIdx]
#         group_speech[pIdx] += speech[pIdx]

#     # if cnt>20:
#     #     break
#     cnt +=1
#     # if cnt>5:
#     #     break
# for pIdx in range(3):
#     group_data[pIdx] = np.array(group_data[pIdx])
#     group_speech[pIdx] = np.array(group_speech[pIdx])
# print("Data shape: {}".format(group_data[0].shape))
# #np.savez_compressed('data_hagglingSellers_speech_face_60frm_5gap_white_training', clips=data_clips, classes=data_speech)
# np.savez('data_hagglingSellers_speech_group_face_30frm_10gap_white_training', clips=group_data, speech=group_speech)




faceParamDir = '/ssd/codes/pytorch_motionSynth/motionsynth_data/data/processed_panoptic/panopticDB_faceMesh_pkl_hagglingProcessed'
"""Haggling training games sellers"""
bTesting = False
featureNum = 5

h36m_files = get_files_haggling(faceParamDir,bTesting)
print('Num: {}'.format(len(h36m_files)))
group_data = [ [], [], [] ]
group_speech = [ [], [], [] ]

cnt =0
for i, item in enumerate(h36m_files):
    print('Processing %i of %i (%s)' % (i, len(h36m_files), item))
    clips, speech = process_file_withSpeech_byGroup(item, 120, 5, featureNum)

    for pIdx in range(3):
        group_data[pIdx] += clips[pIdx]
        group_speech[pIdx] += speech[pIdx]

    # if cnt>20:
    #     break
    cnt +=1
    if cnt>5:
        break
for pIdx in range(3):
    group_data[pIdx] = np.array(group_data[pIdx])
    group_speech[pIdx] = np.array(group_speech[pIdx])
print("Data shape: {}".format(group_data[0].shape))
#np.savez_compressed('data_hagglingSellers_speech_face_60frm_5gap_white_training', clips=data_clips, classes=data_speech)
np.savez('data_hagglingSellers_speech_group_face_120frm_5gap_white_5dim_training_tiny', clips=group_data, speech=group_speech)



# """Haggling training games sellers"""
# h36m_files = get_files_haggling_sellers('panoptic',True)
# print(h36m_files)
# h36m_clips = []
# h36m_classes = []
# for i, item in enumerate(h36m_files):
#     print('Processing %i of %i (%s)' % (i, len(h36m_files), item))
#     clips, speech = process_file_withSpeech(item,30,1)
#     h36m_clips += clips
#     h36m_classes += speech
#     if i==2:
#         break
# data_clips = np.array(h36m_clips)
# data_speech = np.array(h36m_classes)
# np.savez_compressed('data_panoptic_speech_haggling_sellers_testing_byFrame_white_30frm_tiny', clips=data_clips, classes=data_speech)
