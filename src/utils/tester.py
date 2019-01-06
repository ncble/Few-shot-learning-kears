import numpy as np

def create_finetuningset(dataset,way=5,shot=5,querysize=16):
    """
    create train set, val set and test set for fine-tuning stage
    attention: change the labels to 1-...? 先不改试试看
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

    train_img = np.zeros((way * shot, imgs.shape[1], imgs.shape[2], imgs.shape[3]))
    train_label = []
    val_img = np.zeros((way * querysize, imgs.shape[1], imgs.shape[2], imgs.shape[3]))
    val_label = []
    test_img = np.zeros((way * querysize, imgs.shape[1], imgs.shape[2], imgs.shape[3]))
    test_label = []

    for i, label in enumerate(labels_pick):
        print("=================")
        print('class: ', label)
        indices = np.where(labels == label)[0]
        indices_pick = list(np.random.choice(indices, shot + 2 * querysize, replace=True))
        indices_train = indices_pick[:shot]
        indices_val = indices_pick[shot:shot + querysize]
        indices_test = indices_pick[shot + querysize:]

        tr_img = imgs[indices_train]
        va_img = imgs[indices_val]
        te_img = imgs[indices_test]
        tr_label = labels[indices_train]
        va_label = labels[indices_val]
        te_label = labels[indices_test]

        train_img[i * shot:(i + 1) * shot] = tr_img
        train_label += list(tr_label)
        val_img[i * querysize:(i + 1) * querysize] = va_img
        val_label += list(va_label)
        test_img[i * querysize:(i + 1) * querysize] = te_img
        test_label += list(te_label)

    return train_img,train_label,val_img,val_label,test_img,test_label






    pass

def test_model(model,
                x,y
                ):
    model.evaluate(x,y)