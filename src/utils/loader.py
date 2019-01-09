
import os
import pickle

# base_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..')
# data_folder_path = os.path.join(base_path, 'data')

def load_miniImgnet(filepath):
    # MiniImagenet_path = os.path.join(data_folder_path, 'MiniImagenet')
    
    with open(filepath, 'rb') as file:
    	A = pickle.load(file, encoding='bytes')
    return A
