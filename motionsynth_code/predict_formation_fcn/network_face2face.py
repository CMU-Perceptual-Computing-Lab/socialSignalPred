import numpy as np
import os

import modelZoo_face2face
import torch

class Network_face2face():
    def __init__(self):

        checkpointRoot = './face2face/'
        checkpointFolder = checkpointRoot+ '/social_regressor_fcn_bn_encoder_noDrop/'
        preTrainFileName= 'checkpoint_e2550_loss0.7280.pth'
        preTrainFileName= 'checkpoint_e6750_loss0.7233.pth'

        ######################################
        # Load Options
        log_file_name = os.path.join(checkpointFolder, 'opt.json')
        import json
        with open(log_file_name, 'r') as opt_file:
            options_dict = json.load(opt_file)

            ##Setting Options to Args
            model_name = options_dict['model']

        model = getattr(modelZoo_face2face,model_name)().cuda()
        model.eval()

        preprocess = np.load(checkpointFolder + 'preprocess_core.npz') #preprocess['Xmean' or 'Xstd']: (1, 73,1)

        #model = getattr(modelZoo_traj2Body,args.model)().cuda()
        model.eval()

        #Create Model
        trainResultName = checkpointFolder + preTrainFileName
        loaded_state = torch.load(trainResultName, map_location=lambda storage, loc: storage)

        model.load_state_dict(loaded_state,strict=False) #strict False is needed some version issue for batchnorm
        self.model = model.eval()  #Do I need this again?

        self.mean = preprocess['body_mean']
        self.std = preprocess['body_std']

        self.mean_two = preprocess['body_mean_two']
        self.std_two = preprocess['body_std_two']

        ############################################################################
        # Import Pretrained Autoencoder

        ######################################
        # Checkout Folder and pretrain file setting
        ae_checkpointRoot = './'
        ae_checkpointFolder = ae_checkpointRoot+ '/face2face/social_autoencoder_first_try4/'
        preTrainFileName= 'checkpoint_e100_loss0.0351.pth'

        # ######################################
        # # Load Pretrained Auto-encoder
        ae_preprocess = np.load(ae_checkpointFolder + 'preprocess_core.npz') #preprocess['Xmean' or 'Xstd']: (1, 73,1))
        model_ae = modelZoo_face2face.autoencoder_first(5).cuda()

        #Creat Model
        trainResultName = ae_checkpointFolder + preTrainFileName
        loaded_state = torch.load(trainResultName, map_location=lambda storage, loc: storage)

        model_ae.load_state_dict(loaded_state,strict=False) #strict False is needed some version issue for batchnorm
        self.model_ae = model_ae.eval()  #Do I need this again?

    def standardize_input(self,inputData):

        inputData_std = (inputData - self.mean_two) / self.std_two

        return inputData_std

    def destandardize_output(self,output_std):
        output = output_std*self.std + self.mean
        return output


    def __call__(self,inputData):

        output = self.model(inputData)
        output = self.model_ae.decoder(output)

        return output
        
