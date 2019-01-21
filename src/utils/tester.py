import numpy as np

def create_finetuningset(dataset, way=5, shot=1, querysize=16):
    """
    create train set, val set and test set for fine-tuning stage
    attention: change the labels to 0 - classes-1
    :param dataset: test
    :param way:
    :param shot:
    :return: train,val,test
    """
    # pick classes
    catname2label = dataset[b'catname2label']  # a dict
    imgs = dataset[b'data']  # a numpy array
    labels = np.array(dataset[b'labels'])  # a numpy array
    labels_pick = np.random.choice(list(catname2label.values()), way, replace=False)

    train_img = np.zeros((way*shot, imgs.shape[1], imgs.shape[2], imgs.shape[3]))
    train_label = []
    val_img = np.zeros((way*querysize, imgs.shape[1], imgs.shape[2], imgs.shape[3]))
    val_label = []
    test_img = np.zeros((way*querysize, imgs.shape[1], imgs.shape[2], imgs.shape[3]))
    test_label = []
    
    for i, label in enumerate(labels_pick):
        # print("=================")
        # print('class: ', label)
        indices = np.where(labels == label)[0]
        # indices_pick = list(np.random.choice(indices, shot+2*querysize, replace=True))
        indices_pick = list(np.random.choice(indices, shot+2*querysize, replace=False)) # TODO?
        indices_train = indices_pick[:shot]
        indices_val = indices_pick[shot:shot + querysize]
        indices_test = indices_pick[shot + querysize:]

        tr_img = imgs[indices_train]
        va_img = imgs[indices_val]
        te_img = imgs[indices_test]
        # tr_label = labels[indices_train]
        tr_label = [i for j in range(shot)]
        # va_label = labels[indices_val]
        va_label = [i for j in range(querysize)]
        # te_label = labels[indices_test]
        te_label = [i for j in range(querysize)]

        train_img[i*shot:(i+1)*shot] = tr_img
        train_label += tr_label
        #.tolist()
        val_img[i*querysize:(i+1)*querysize] = va_img
        val_label += va_label
        #.tolist()
        test_img[i*querysize:(i+1)*querysize] = te_img
        test_label += te_label
        #.tolist()

    return train_img, train_label, val_img, val_label, test_img, test_label, labels_pick

if __name__ == "__main__":
    print("Start")
    import os
    import loader

    test_path = os.path.join("../../data/MiniImagenet", 'miniImageNet_category_split_test.pickle')
    test_set = loader.load_miniImgnet(test_path)
    A = create_finetuningset(test_set)
    import ipdb; ipdb.set_trace()


