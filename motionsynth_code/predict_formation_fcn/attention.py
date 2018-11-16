import numpy as np
import sys

sys.path.append('../../motionsynth_data/motion')
from Quaternions import Quaternions
from Pivots import Pivots

from sklearn.preprocessing import normalize

#input 2xframes
#output 3xframes where Yaxis has zeros
def data_2dTo3D(data_2d, newRowIdx =1):
    data_3d = np.zeros((3, data_2d.shape[1])) #(3,frames)

    if newRowIdx==0:
        data_3d[1,:] = data_2d[0,:]
        data_3d[2,:] = data_2d[1,:]
    elif newRowIdx==1:
        data_3d[0,:] = data_2d[0,:]
        data_3d[2,:] = data_2d[1,:]
    else:#elif newRowIdx==2:
        data_3d[0,:] = data_2d[0,:]
        data_3d[1,:] = data_2d[1,:]

    return data_3d

""" 
    Convert attention to direction

    input: 
        - targetPos: (3, frame)
        - rightPos: (3, frame)
        - leftPos: (3, frame)
        - binaryAtten (frames,) where each value is 0 (right) to 1 (left)
"""
def attention2Direction(targetPos, rightPos, leftPos, binaryAtten):

    if targetPos.shape[0]==2:
        targetPos = data_2dTo3D(targetPos)

    if rightPos.shape[0]==2:
        rightPos = data_2dTo3D(rightPos)
    
    if leftPos.shape[0]==2:
        leftPos = data_2dTo3D(leftPos)
    
    binaryAtten = binaryAtten[:targetPos.shape[1]]

    pos2right = rightPos - targetPos  #me->right (l->b): 0
    pos2left = leftPos - targetPos  

    pos2right = np.swapaxes(normalize(pos2right,axis=0),0,1)    #(frame,3)
    pos2left = np.swapaxes(normalize(pos2left,axis=0),0,1)    #(frame,3)

    rotation_right2left = Quaternions.between(pos2right, pos2left)[:,np.newaxis]     #rot from 0->1
    angle_right2left = Pivots.from_quaternions(rotation_right2left).ps      #(frame,1)

    angleValue_right2FaceNormal = binaryAtten * np.squeeze(angle_right2left)       #This much angle from pos2Other_0

    eurlers_right2FaceNormal = np.zeros( (len(angleValue_right2FaceNormal), 3))        #(frame,3)
    eurlers_right2FaceNormal[:,1] = angleValue_right2FaceNormal        #(frame,3)

    rotation_right2FaceNormal = Quaternions.from_euler(eurlers_right2FaceNormal)    #(frame,)
    faceNormal_byAtten = rotation_right2FaceNormal * pos2right     #(frame,3)

    faceNormal_byAtten = np.swapaxes(faceNormal_byAtten,0,1)    #(3,frame) 

    return faceNormal_byAtten
    