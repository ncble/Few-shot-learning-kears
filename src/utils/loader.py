
import os
import pickle
import numpy as np
# base_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..')
# data_folder_path = os.path.join(base_path, 'data')

def load_miniImgnet(filepath):
    # MiniImagenet_path = os.path.join(data_folder_path, 'MiniImagenet')
    
    with open(filepath, 'rb') as file:
        A = pickle.load(file, encoding='bytes')
    return A
def load_fewshot_testset(X, Y, way=5, shot=1, querysize=16):
    """
    create train set, val set and test set for fine-tuning stage
    attention: change the labels to 0 - classes-1
    :param dataset: test
    :param way:
    :param shot:
    :return: train,val,test
    """
    # pick classes
    all_labels = np.unique(Y)

    labels_pick = np.random.choice(all_labels, way, replace=False)

    train_img = np.zeros((way*shot,)+X.shape[1:])
    train_label = np.tile(np.arange(way)[:, None, None], (shot, 1)).reshape(way*shot, 1)
    val_img = np.zeros((way*querysize,)+X.shape[1:])
    val_label = np.tile(np.arange(way)[:, None, None], (querysize, 1)).reshape(way*querysize, 1)
    test_img = np.zeros((way*querysize,)+X.shape[1:])
    test_label = np.tile(np.arange(way)[:, None, None], (querysize, 1)).reshape(way*querysize, 1)
    
    for i, label in enumerate(labels_pick):
        # print("=================")
        # print('class: ', label)
        indices = np.where(Y == label)[0]
        # indices_pick = list(np.random.choice(indices, shot+2*querysize, replace=True))
        indices_pick = np.random.choice(indices, shot+2*querysize, replace=False) # TODO?
        indices_train = indices_pick[:shot]
        indices_val = indices_pick[shot:shot + querysize]
        indices_test = indices_pick[shot + querysize:]

        tr_img = X[indices_train]
        va_img = X[indices_val]
        te_img = X[indices_test]
        
        train_img[i*shot:(i+1)*shot] = tr_img
        val_img[i*querysize:(i+1)*querysize] = va_img
        test_img[i*querysize:(i+1)*querysize] = te_img
        

    return train_img, train_label, val_img, val_label, test_img, test_label, labels_pick