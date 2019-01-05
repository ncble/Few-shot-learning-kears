
import os
import pickle

base_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..','..')
data_folder_path = os.path.join(base_path,'data')

def load_miniImgnet(filename):
    MiniImagenet_path = os.path.join(data_folder_path,'MiniImagenet')
    return pickle.load(open(os.path.join(MiniImagenet_path,filename),'rb'),encoding='bytes')
