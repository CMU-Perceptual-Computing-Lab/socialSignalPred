import os
import sys
import numpy as np
import scipy.io as io

"""For logging by tensorboard"""
bLog = False  
try:
	sys.path.append('../utils')
	from logger import Logger
	bLog = True
except ImportError:
	pass

import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST
import os
import argparse

import modelZoo

def print_options(parser,opt):
    option_dict = dict()

    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        option_dict[k] = v

        comment = ''
        default = parser.get_default(k)
        if v != default:
            comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)

    return message, option_dict

import json
def save_options(checkpoints_dir, message, options_dict):

    # save to the disk
    #expr_dir = os.path.join(checkpoints_dir, 'options.txt')
    #util.mkdirs(expr_dir)
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)
    file_name = os.path.join(checkpoints_dir, 'opt.txt')
    with open(file_name, 'wt') as opt_file:
        opt_file.write(message)
        opt_file.write('\n')

    file_name = os.path.join(checkpoints_dir, 'opt.json')
    with open(file_name, 'wt') as opt_file:
        json.dump(options_dict,opt_file)
        
parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--epochs', type=int, default=5001, metavar='N',
                    help='number of epochs to train (default: 5001)')


parser.add_argument('--batch', type=int, default=2048, metavar='N',
                    help='batch size (default: 2048)')


parser.add_argument('--gpu', type=int, default=0, metavar='N',
                    help='Select gpu (default: 0)')

parser.add_argument('--checkpoint_freq', type=int, default=50, metavar='N',
                    help='How frequently save the checkpoint (default: every 50 epoch)')

parser.add_argument('--model', type=str, default='autoencoder_first',
                    help='a model name in the model_zoo.py (default: autoencoder_first')

parser.add_argument('--solver', type=str, default='adam_ams',
                    help='Optimization solver. adam or sgd, adam_ams. (default: adam_ams')

parser.add_argument('--db', type=str, default='cmu',
                    help='Database for training cmu...(default: cmu')

parser.add_argument('--finetune', type=str, default='',
                    help='if a folder is specified, then restart training by loading the last saved model files')

parser.add_argument('--check_root', type=str, default='./',
                    help='The root dir to make subfolders for the check point (default: ./) ')


parser.add_argument('--weight_kld', type=float, default='1.0',
                    help='Weight for the KLD term in VAE training (default: 1.0) ')



args = parser.parse_args()  


#Debug
#args.model = 'autoencoder_2convLayers'

#args.model ='autoencoder_3convLayers_vect'
#args.model ='autoencoder_3conv_vae'
#args.finetune = 'autoencoder_3conv_vae_weight'
#args.check_root = '/posefs2b/Users/hanbyulj/pytorch_motionSynth/checkpoint'
#args.weight_kld = 0.01

torch.cuda.set_device(args.gpu)

rng = np.random.RandomState(23456)
torch.manual_seed(23456)
torch.cuda.manual_seed(23456)

"""All available dataset"""
#datapath ='/ssd/codes/pytorch_motionSynth/motionsynth_data' 
datapath ='../../motionsynth_data/data/processed/' 


if args.db == 'cmu':
    #dblist = ['data_cmu']
    dblist = ['data_mhad']
elif args.db == 'holdenAll':
    dblist = ['data_cmu', 'data_hdm05', 'data_mhad', 'data_edin_locomotion', 'data_edin_xsens',
            'data_edin_misc', 'data_edin_punching']
elif args.db == 'human36m_train':
    dblist = ['data_h36m_training']
elif args.db == 'holden_human36m':
    dblist = ['data_cmu', 'data_hdm05', 'data_mhad', 'data_edin_locomotion', 'data_edin_xsens',
            'data_edin_misc', 'data_edin_punching','data_h36m_training']
else:
    assert(False)
#Xcmu = np.load(datapath +'/data/processed/data_cmu.npz')['clips'] # (17944, 240, 73)
# Xhdm05 = np.load(datapath +'/data/processed/data_hdm05.npz')['clips']	#(3190, 240, 73)
# Xmhad = np.load(datapath +'/data/processed/data_mhad.npz')['clips'] # (2674, 240, 73)
# #Xstyletransfer = np.load('/data/processed/data_styletransfer.npz')['clips']
# Xedin_locomotion = np.load(datapath +'/data/processed/data_edin_locomotion.npz')['clips'] #(351, 240, 73)
# Xedin_xsens = np.load(datapath +'/data/processed/data_edin_xsens.npz')['clips'] #(1399, 240, 73)
# Xedin_misc = np.load(datapath +'/data/processed/data_edin_misc.npz')['clips'] #(122, 240, 73)
# Xedin_punching = np.load(datapath +'/data/processed/data_edin_punching.npz')['clips'] #(408, 240, 73)
#h36m_training = np.load(datapath +'/data/processed/data_h36m_training.npz')['clips'] #(13156, 240, 73)

db_loaded =list()
for dbname in dblist:
    X_temp = np.load(datapath + dbname + '.npz')['clips'] 
    db_loaded.append(X_temp)
X = np.concatenate(db_loaded, axis=0)


""" Training Network """
num_epochs = args.epochs#500
#batch_size = 128
batch_size = args.batch
learning_rate = 1e-3

model = getattr(modelZoo,args.model)().cuda()

for param in model.parameters():
    print(type(param.data), param.size())

criterion = nn.MSELoss()
if args.solver == 'adam':
    print('solver: Adam')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
elif args.solver == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    print('solver: SGD')
elif args.solver == 'adam_ams': #only for pytorch 0.4 or later.
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5, amsgrad=True)
    print('solver: Adam with AMSGrad')
else:
    print('Unknown solver option')
    assert(False)

#checkpointFolder = './autoenc_vect/'
pretrain_epoch = 0
if args.finetune =='':
    pretrain_batch_size =args.batch  #Assume current batch was used in pretraining
    checkpointFolder = args.check_root + '/'+ model.__class__.__name__
    if not os.path.exists(checkpointFolder):
        os.mkdir(checkpointFolder)
    else: #if already exist
        tryIdx =1
        while True:
            newCheckName = checkpointFolder + '_try' + str(tryIdx)
            if not os.path.exists(newCheckName):
                checkpointFolder = newCheckName
                os.mkdir(checkpointFolder)
                break
            else:
                tryIdx += 1
else:       #FineTuning
    checkpointFolder = args.check_root+ '/' + args.finetune

    checkpointList =  [os.path.join(checkpointFolder,f) for f in sorted(list(os.listdir(checkpointFolder)))
                if os.path.isfile(os.path.join(checkpointFolder,f))
                    and f.endswith('.pth') ] 

    #Fine Last Epoch
    last_epoch =0
    for name in checkpointList:
        pretrain_epoch= int(name[(name.find('checkpoint_') + 12):-4])
        if last_epoch <pretrain_epoch:
            last_epoch= pretrain_epoch
            trainResultName = name


    #checkpointList.sort(key=lambda x: os.path.getmtime(x))
    #trainResultName =checkpointList[-1]
    pretrain_epoch= int(trainResultName[(trainResultName.find('checkpoint_') + 12):-4])
    pretrain_epoch = pretrain_epoch+1
    
    try:
        log_file_name = os.path.join(checkpointFolder, 'opt.json')
        with open(log_file_name, 'r') as opt_file:
            options_dict = json.load(opt_file)
        pretrain_batch_size = options_dict['batch']

    except:
        pretrain_batch_size =args.batch  #Assume current batch was used in pretraining
        
    #Fine Last Epoch file
    print('load previous state: {0}'.format(trainResultName))
    model.load_state_dict(torch.load(trainResultName, map_location=lambda storage, loc: storage),strict=False)

    #Checking optimizer file
    trainResultName = trainResultName+'o'
    if os.path.exists(trainResultName):
        print('load previous state: {0}'.format(trainResultName))
        optimizer.load_state_dict(torch.load(trainResultName, map_location=lambda storage, loc: storage))

    model.train()
    model.eval()


if bLog:
    logger = Logger(checkpointFolder+'/logs')



""" Save Option Info """
option_str, options_dict = print_options(parser,args)

option_str += '\nDBList: \n'
for i, dbname in enumerate(dblist):
    option_str +=  '{0}:  {1}\n'.format(dbname,db_loaded[i].shape)
option_str += 'All: {0}'.format(X.shape)
options_dict['dblist']= dblist

save_options(checkpointFolder, option_str, options_dict)


""" Compute mean and std """
X = np.swapaxes(X, 1, 2).astype(np.float32)
feet = np.array([12,13,14,15,16,17,24,25,26,27,28,29])

Xmean = X.mean(axis=2).mean(axis=0)[np.newaxis,:,np.newaxis]
Xmean[:,-7:-4] = 0.0
Xmean[:,-4:]   = 0.5

Xstd = np.array([[[X.std()]]]).repeat(X.shape[1], axis=1)
Xstd[:,feet]  = 0.9 * Xstd[:,feet]
Xstd[:,-7:-5] = 0.9 * X[:,-7:-5].std()
Xstd[:,-5:-4] = 0.9 * X[:,-5:-4].std()
Xstd[:,-4:]   = 0.5

""" Data standardization """
X = (X - Xmean) / Xstd
I = np.arange(len(X))
rng.shuffle(I); X = X[I]
print('Input data size: {0}'.format(X.shape))

np.savez_compressed(checkpointFolder+'/preprocess_core.npz', Xmean=Xmean, Xstd=Xstd)

#stepNum =0



#Compute stepNum start point (to be continuos in tensorboard if pretraine data is loaded)
stepNum = pretrain_epoch* len(np.arange(X.shape[0] // pretrain_batch_size))
for epoch in range(num_epochs):

    batchinds = np.arange(X.shape[0] // batch_size)
    rng.shuffle(batchinds)
    
    for bii, bi in enumerate(batchinds):

        idxStart  = bi*batch_size
        inputData = X[idxStart:(idxStart+batch_size),:,:]      #Huge bug!!
        #inputData = X[bi:(bi+batch_size),:,:]      #Huge bug!!
        inputData = Variable(torch.from_numpy(inputData)).cuda()

        if "vae" in args.model:
            # ===================forward=====================
            output, mu, logvar = model(inputData)
            #loss = criterion(output, inputData)
            #loss = modelZoo.vae_loss_function(output, inputData, mu, logvar,criterion)
            loss,recon_loss,kld_loss = modelZoo.vae_loss_function(output, inputData, mu, logvar,criterion,args.weight_kld)
            

            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ===================log========================
            print('model: {}, epoch [{}/{}], loss:{:.4f} (recon: {:.4f}, kld {:.4f})'
                        .format(args.model, epoch +pretrain_epoch, num_epochs, loss.item(), recon_loss.item(), kld_loss.item()))

        else:
             # ===================forward=====================
            output = model(inputData)
            loss = criterion(output, inputData)

            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ===================log========================
            if int(torch.__version__[2])==2:
                print('model: {}, epoch [{}/{}], loss:{:.4f}'
                        .format(args.model, epoch + pretrain_epoch, num_epochs, loss.data[0]))
            else:
                print('model: {}, epoch [{}/{}], loss:{:.4f}'
                        .format(args.model, epoch +pretrain_epoch, num_epochs, loss.item()))
        

        if bLog:
            # 1. Log scalar values (scalar summary)
            if int(torch.__version__[2])==2:
                info = { 'loss': loss.data[0] }
            else:
                info = { 'loss': loss.item() }

            for tag, value in info.items():
                logger.scalar_summary(tag, value, stepNum)
        
        stepNum = stepNum+1

        # # 2. Log values and gradients of the parameters (histogram summary)
        # for tag, value in model.named_parameters():
        #     tag = tag.replace('.', '/')
        #     logger.histo_summary(tag, value.data.cpu().numpy(), epoch+1)
        #     logger.histo_summary(tag+'/grad', value.grad.data.cpu().numpy(), epoch+1)
    

    # #Compute Error Again here.
    # # batchinds = np.arange(X.shape[0] // batch_size)
    # test_loss = 0
    # for bii, bi in enumerate(batchinds):
    #    # print('{} /{}\n'.format(bii,len(batchinds)))
    #     Xorgi_stdd = X[bi:(bi+batch_size),:,:]  #Input (batchSize,73,240) 
    #     Xrecn = model(Variable(torch.from_numpy(Xorgi_stdd)).cuda())
    #     loss = criterion(Xrecn, Variable(torch.from_numpy(Xorgi_stdd)).cuda())
    #     test_loss += loss.data.cpu().numpy().item()* batch_size # sum up batch loss   
    # test_loss /= len(batchinds)*batch_size
    # print('Testing: epoch:{} /Average loss: {:.4f}\n'.format(epoch +1, test_loss))
    # #     test_loss, correct, len(test_loader.dataset),
    # #     100. * correct / len(test_loader.dataset)))
 
    if (epoch + pretrain_epoch) % args.checkpoint_freq == 0:
    #if (epoch + pretrain_epoch) % 10 == 0:
        fileName = checkpointFolder+ '/checkpoint_e' + str(epoch + pretrain_epoch) + '.pth'
        torch.save(model.state_dict(), fileName)
        fileName = checkpointFolder+ '/checkpoint_e' + str(epoch + pretrain_epoch) + '.ptho'
        torch.save(optimizer.state_dict(), fileName)
        #torch.save(model, fileName)
