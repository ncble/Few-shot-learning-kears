import os
from keras.callbacks import ModelCheckpoint

base_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..','..')
weights_folder_path = os.path.join(base_path,'weights')


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

def train_model(model,
                x,y,
                shuffle=True,
                val_data=None ,
                batch_size=16,
                epochs=400 ,
                foldername=None, # the folder under the dir weithts, for saveing weights
                weights_file=None , # save best weights
                monitor='val_acc',
                ):
    if weights_file is None:
        print('Error! The filename for weights is empty')
        history = -1
    else:
        model_folder_path = os.path.join(weights_folder_path ,foldername)

        save_model_path = check_savepath(foldpath=model_folder_path , filename=weights_file)

        history = model.fit(x=x,y=y,
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_data=val_data,
                            shuffle=shuffle,
                            callbacks=[ModelCheckpoint(save_model_path,
                                                       monitor=monitor,
                                                       save_best_only=True)]
                            )

    return history