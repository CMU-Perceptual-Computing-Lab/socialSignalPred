
import pickle


fileName = '/ssd/data/neutrMosh/neutrSMPL_CMU/01/01_01.pkl'

with open(fileName, 'rb') as f:
    data = pickle.load(f)


print(data)


