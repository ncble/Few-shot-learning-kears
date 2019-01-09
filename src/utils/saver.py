import os
import pickle

# base_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..')
# output_folder_path = os.path.join(base_path,'output')

def check_dirpath(filepath):
    """
    check if the path to store files exists, and remove the existing files
    :param foldpath: the path of folder which stores the file
    :param filename: the name of the file
    :return: the absolute path of the file to save
    """
    dirpath = "/".join(filepath.split("/")[:-1])
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
        
def save_output(obj, filepath):
    check_dirpath(filepath)
    with open(filepath, 'wb') as file:
        pickle.dump(obj, file)

