import os
import pickle

base_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..','..')
output_folder_path = os.path.join(base_path,'output')

def save_output(obj,filename):
    return pickle.dump(obj,open(os.path.join(output_folder_path,filename),'rb'))
