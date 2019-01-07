import os
import pickle

base_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..','..')
output_folder_path = os.path.join(base_path,'output')


def check_savepath(foldpath, filename):
    """
    check if the path to store files exists, and remove the existing files
    :param foldpath: the path of folder which stores the file
    :param filename: the name of the file
    :return: the absolute path of the file to save
    """
    if not os.path.exists(foldpath):
        os.makedirs(foldpath)
    save_path = os.path.join(foldpath, filename)
    try:
        os.remove(save_path)
    except OSError:
        pass
    return save_path

def save_output(obj,foldername,filename):
    save_folder_path = os.path.join(output_folder_path, foldername)

    save_file_path = check_savepath(foldpath=save_folder_path, filename=filename)
    return pickle.dump(obj,open(save_file_path,'wb'))
