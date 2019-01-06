import os
from keras.callbacks import ModelCheckpoint

base_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..','..')
weights_folder_path = os.path.join(base_path,'weights')

def train_model(model,
                x,y,
                shuffle=True,
                val_data=None ,
                batch_size=16,
                epochs=400 ,
                weights_file=None ,
                monitor='val_acc',
                ):
    if weights_file is None:
        print('Error! The filename for weights is empty')
        history = -1
    else:
        save_model_path = os.path.join(weights_folder_path ,weights_file)

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