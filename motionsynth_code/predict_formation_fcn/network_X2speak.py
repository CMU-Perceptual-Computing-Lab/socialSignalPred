import numpy as np
import os

#import modelZoo as modelZoo_Xspeak
import torch

import modelZoo_Xspeak as modelZoo_Xspeak

class Network_facebody2speak():
    def __init__(self, bOwnBody= True):

        if bOwnBody:
            #All signals
            checkpointRoot = '/posefs2b/Users/hanbyulj/pytorch_motionSynth/checkpoint/thesis_speack_classifcation/byOwnbody/'
            checkpointFolder = checkpointRoot+ '/social_speackClass_allSignal_all/'
            preTrainFileName= 'checkpoint_e91_acc0.8913.pth'

        else: #Another Seller
            checkpointRoot = '/posefs2b/Users/hanbyulj/pytorch_motionSynth/checkpoint/thesis_speack_classifcation/byOtherSeller/'
            checkpointFolder = checkpointRoot+ '/social_speackClass_allSignal_try12_allSignal/'
            preTrainFileName= 'checkpoint_e124_acc0.7797.pth'


        

        ######################################
        # Load Options
        log_file_name = os.path.join(checkpointFolder, 'opt.json')
        import json
        with open(log_file_name, 'r') as opt_file:
            options_dict = json.load(opt_file)

            ##Setting Options to Args
            model_name = options_dict['model']

        print('Loading model: {}'.format(model_name))
        model = getattr(modelZoo_Xspeak,model_name)().cuda()
        model.eval()

        preprocess = np.load(checkpointFolder + 'preprocess_core.npz') #preprocess['Xmean' or 'Xstd']: (1, 73,1)

        #model = getattr(modelZoo_traj2Body,args.model)().cuda()
        model.eval()

        #Create Model
        trainResultName = checkpointFolder + preTrainFileName
        loaded_state = torch.load(trainResultName, map_location=lambda storage, loc: storage)

        model.load_state_dict(loaded_state,strict=False) #strict False is needed some version issue for batchnorm
        self.model = model.eval()  #Do I need this again?

        self.Xmean = preprocess['Xmean']
        self.Xstd = preprocess['Xstd']


    #inputData: (batch,featureDim,frames)
    def standardize_input(self,inputData):

        inputData_std = (inputData - self.Xmean) / self.Xstd

        return inputData_std

    # def destandardize_output(self,output_std):
    #     output = output_std*self.body_std + self.body_mean
    #     return output

    #inputData: (batch,featureDim,frames)
    def __call__(self,inputData):

        output = self.model(inputData)

        return output
        


class Network_body2speak():
    def __init__(self, bOwnBody= True):

        if bOwnBody:
            #All signals
            checkpointRoot = '/posefs2b/Users/hanbyulj/pytorch_motionSynth/checkpoint/thesis_speack_classifcation/byOwnbody/'
            checkpointFolder = checkpointRoot+ '/social_speackClass_allSignal_try1_bodyOnly/'
            preTrainFileName= 'checkpoint_e72_acc0.7669.pth'

        else: #Another Seller
            checkpointRoot = '/posefs2b/Users/hanbyulj/pytorch_motionSynth/checkpoint/thesis_speack_classifcation/byOtherSeller/'
            checkpointFolder = checkpointRoot+ '/social_speackClass_allSignal_try3_bodyOnly/'
            preTrainFileName= 'checkpoint_e30_acc0.7129.pth'


        

        ######################################
        # Load Options
        log_file_name = os.path.join(checkpointFolder, 'opt.json')
        import json
        with open(log_file_name, 'r') as opt_file:
            options_dict = json.load(opt_file)

            ##Setting Options to Args
            model_name = options_dict['model']

        print('Loading model: {}'.format(model_name))
        model = getattr(modelZoo_Xspeak,model_name)().cuda()
        model.eval()

        preprocess = np.load(checkpointFolder + 'preprocess_core.npz') #preprocess['Xmean' or 'Xstd']: (1, 73,1)

        #model = getattr(modelZoo_traj2Body,args.model)().cuda()
        model.eval()

        #Create Model
        trainResultName = checkpointFolder + preTrainFileName
        loaded_state = torch.load(trainResultName, map_location=lambda storage, loc: storage)

        model.load_state_dict(loaded_state,strict=False) #strict False is needed some version issue for batchnorm
        self.model = model.eval()  #Do I need this again?

        self.Xmean = preprocess['Xmean']
        self.Xstd = preprocess['Xstd']


    #inputData: (batch,featureDim,frames)
    def standardize_input(self,inputData):

        inputData_std = (inputData - self.Xmean) / self.Xstd

        return inputData_std

    # def destandardize_output(self,output_std):
    #     output = output_std*self.body_std + self.body_mean
    #     return output

    #inputData: (batch,featureDim,frames)
    def __call__(self,inputData):

        output = self.model(inputData)

        return output
        


class Network_face2speak():
    def __init__(self, bOwnBody= True):

        if bOwnBody:
            #All signals
            checkpointRoot = '/posefs2b/Users/hanbyulj/pytorch_motionSynth/checkpoint/thesis_speack_classifcation/byOwnbody/'
            checkpointFolder = checkpointRoot+ '/social_speackClass_allSignal_try2_faceOnly/'
            preTrainFileName= 'checkpoint_e60_acc0.8916.pth'

        else: #Another Seller
            checkpointRoot = '/posefs2b/Users/hanbyulj/pytorch_motionSynth/checkpoint/thesis_speack_classifcation/byOtherSeller/'
            checkpointFolder = checkpointRoot+ '/social_speackClass_allSignal_try4_faceOnly/'
            preTrainFileName= 'checkpoint_e57_acc0.8021.pth'


        ######################################
        # Load Options
        log_file_name = os.path.join(checkpointFolder, 'opt.json')
        import json
        with open(log_file_name, 'r') as opt_file:
            options_dict = json.load(opt_file)

            ##Setting Options to Args
            model_name = options_dict['model']

        print('Loading model: {}'.format(model_name))
        model = getattr(modelZoo_Xspeak,model_name)().cuda()
        model.eval()

        preprocess = np.load(checkpointFolder + 'preprocess_core.npz') #preprocess['Xmean' or 'Xstd']: (1, 73,1)

        #model = getattr(modelZoo_traj2Body,args.model)().cuda()
        model.eval()

        #Create Model
        trainResultName = checkpointFolder + preTrainFileName
        loaded_state = torch.load(trainResultName, map_location=lambda storage, loc: storage)

        model.load_state_dict(loaded_state,strict=False) #strict False is needed some version issue for batchnorm
        self.model = model.eval()  #Do I need this again?

        self.Xmean = preprocess['Xmean']
        self.Xstd = preprocess['Xstd']


    #inputData: (batch,featureDim,frames)
    def standardize_input(self,inputData):

        inputData_std = (inputData - self.Xmean) / self.Xstd

        return inputData_std

    # def destandardize_output(self,output_std):
    #     output = output_std*self.body_std + self.body_mean
    #     return output

    #inputData: (batch,featureDim,frames)
    def __call__(self,inputData):

        output = self.model(inputData)

        return output
        
