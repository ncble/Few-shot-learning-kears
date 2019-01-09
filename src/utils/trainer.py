import os
from keras.callbacks import ModelCheckpoint

# base_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..')
# weights_dirpath = os.path.join(base_path, 'weights')


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

    # return save_path

def train_model(model,
                x, y,
                shuffle=True,
                val_data=None ,
                batch_size=16,
                epochs=400 ,
                # foldername=None, # the folder under the dir weithts, for saveing weights
                # weights_file=None , # save best weights
                monitor='val_acc',
                save_weights_path=None
                # weights_dirpath=None
                ):
    if save_weights_path is None:
        print('Error! The filename for weights is empty')
        history = -1
    else:
        # model_folder_path = os.path.join(weights_dirpath, foldername)

        # save_model_path = check_dirpath(foldpath=model_folder_path, filename=weights_file)
        check_dirpath(save_weights_path)

        history = model.fit(x=x, y=y,
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_data=val_data,
                            shuffle=shuffle,
                            callbacks=[ModelCheckpoint(save_weights_path,
                                                       monitor=monitor,
                                                       save_best_only=True)]
                            )

    return history.history