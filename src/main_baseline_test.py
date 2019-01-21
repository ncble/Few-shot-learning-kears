import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--gpu_num", help="Choose the number of GPU.", type=str, default="0", choices=["0", "1"])
parser.add_argument("--gpu_fra", help="Decide the GPU fraction to use.", type=float, default=0.95)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_num
gpu_fraction = args.gpu_fra

import keras
import keras.backend as K
import tensorflow as tf
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
K.set_session(sess)

from utils import trainer, loader, tester, saver
from models.custom_models import Baseline_plus, Baseline
#from src.models.backbones import ResNet10
from keras.optimizers import Adam
import pickle
import numpy as np

####################### training stage ###################
# The training stage is for training backbone
#########################################################
# the weights will be saved at 'weights/training_setting/'
# the output will be saved at 'output/training_setting/'
# if 'training_setting' does not exist under 'weights/' or 'output/', it will be created automatically


#######################
##      Config
#######################



#######################
"""
# load image
# train_train_path = 'miniImageNet_category_split_train_phase_train.pickle'
# train_val_path = 'miniImageNet_category_split_train_phase_val.pickle'

train_train = loader.load_miniImgnet(train_train_path) 
train_val = loader.load_miniImgnet(train_val_path)

train_img = train_train[b'data'] ## shape = (38400, 84, 84, 3)
train_label = train_train[b'labels']

val_img = train_val[b'data'] ## shape = (18748, 84, 84, 3)
val_label = train_val[b'labels']

# define input shape, classes
input_shape = train_img.shape[1:]
classes = len(train_train[b'catname2label'])

train_img = train_img/255.
val_img = val_img/255.


# baseline
baseline = Baseline(input_shape=input_shape, classes=classes)
baseline.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=1e-3), 
              metrics=['accuracy'])

history_baseline = trainer.train_model(model=baseline,
                    x=train_img, y=train_label,
                    shuffle=True,
                    val_data=(val_img, val_label),
                    batch_size=64,
                    epochs=train_epoch ,
                    # foldername=EXP_NAME,
                    # weights_file='baseline_training.h5', # save weights
                    save_weights_path=os.path.join("../weights/{}/{}".format(EXP_NAME, \
                                        'baseline_training.h5')),
                    monitor='val_acc',
                    )
out_filepath = os.path.join("../output/{}/{}".format(EXP_NAME, 'history_baseline_training.pickle'))
saver.save_output(history_baseline, out_filepath)

# baseline_plus
baseline_plus = Baseline_plus(input_shape=input_shape, classes=classes)
baseline_plus.compile(loss='sparse_categorical_crossentropy',
                      optimizer=Adam(lr=1e-3),
                      metrics=['accuracy'])
history_baseline_plus = trainer.train_model(model=baseline_plus,
                x=train_img, y=train_label,
                shuffle=True,
                val_data=(val_img, val_label),
                batch_size=64,
                epochs=train_epoch ,
                # foldername=EXP_NAME,
                # weights_file='baseline_plus_training.h5', # save weights
                save_weights_path=os.path.join("../weights/{}/{}".format(EXP_NAME, \
                                        'baseline_plus_training.h5')),
                monitor='val_acc',
                )
out_filepath = os.path.join("../output/{}/{}".format(EXP_NAME, 'history_baseline_plus_training.pickle'))
saver.save_output(history_baseline_plus, out_filepath)

# release the variables
del train_train, train_val, train_img, train_label, val_img, val_label
del baseline, baseline_plus, history_baseline, history_baseline_plus

"""

# ##################### fine-tuning stage ######################
# for training a new classifier for novel classes
################################################################
# the weights will be saved at 'weights/finetuning_setting/'
# the output will be saved at 'output/finetuning_setting/'
# if 'finetuning_setting' does not exist under 'weights/' or 'output/', it will be created automatically
# save_finetune_dirname = EXP_NAME_fine_tune #'finetuning_setting'
# loadbb_experiment = EXP_NAME#'training_setting'

def evaluate(test_path, 
    epochs=100, 
    shot=1,
    way=5,
    querysize=32, 
    verbose=0, 
    load_weights_dirname=None,
    save_finetune_dirname=None
    ):
    test_set = loader.load_miniImgnet(test_path)
    # create dataset for training, validation, testing
    train_img, train_label, val_img, val_label, test_img, test_label, labels_pick  = \
        tester.create_finetuningset(dataset=test_set, way=way, shot=shot, querysize=querysize)
    train_img = train_img/255.
    val_img = val_img/255.
    test_img = test_img/255.


    input_shape = train_img.shape[1:]

    # baseline
    baseline = Baseline(input_shape=input_shape,
                        classes=way,
                        loadbb=True,
                        load_weigths_path=os.path.join("../weights/{}/{}".format(load_weights_dirname, \
                                    'baseline_training.h5')),
                        freezebb=True)
    baseline.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=1e-3),
                  metrics=['accuracy'])

    history_baseline = trainer.train_model(model=baseline,
                    x=train_img, y=train_label,
                    shuffle=True,
                    val_data=(val_img, val_label),
                    batch_size=querysize,
                    epochs=epochs ,
                    save_weights_path=os.path.join("../weights/{}/{}".format(save_finetune_dirname, \
                        'baseline_finetuning_{}shot.h5'.format(shot))),
                    monitor='val_acc',
                    verbose=verbose
                    )
   
    # load best model and 
    baseline.load_weights('../weights/{}/baseline_finetuning_{}shot.h5'.format(save_finetune_dirname, shot))
    performance_baseline_finetune = baseline.evaluate(x=test_img, y=test_label)
    
    # baseline plus
    baseline_plus = Baseline_plus(input_shape=input_shape,
                        classes=way,
                        loadbb=True,
                        load_weigths_path=os.path.join("../weights/{}/{}".format(load_weights_dirname, \
                                    'baseline_plus_training.h5')),
                        freezebb=True)
    baseline_plus.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=1e-3),
                  metrics=['accuracy'])

    history_baseline_plus = trainer.train_model(model=baseline_plus,
                    x=train_img, y=train_label,
                    shuffle=True,
                    val_data=(val_img, val_label),
                    batch_size=querysize,
                    epochs=epochs,
                    save_weights_path=os.path.join("../weights/{}/{}".format(save_finetune_dirname, \
                        'baseline_plus_finetuning_{}shot.h5'.format(shot))),
                    monitor='val_acc',
                    verbose=verbose
                    )
   
    # load best model and 
    baseline_plus.load_weights('../weights/{}/baseline_plus_finetuning_{}shot.h5'.format(save_finetune_dirname, shot))
    performance_baseline_plus_finetune = baseline_plus.evaluate(x=test_img, y=test_label)
   
    dirpath = "../results/{}".format(save_finetune_dirname)
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    with open(os.path.join(dirpath, "result_baseline.txt"), "a") as file:
        np.savetxt(file, np.array(performance_baseline_finetune).reshape(1, -1))
    with open(os.path.join(dirpath, "result_baseline_plus.txt"), "a") as file:
        np.savetxt(file, np.array(performance_baseline_plus_finetune).reshape(1, -1))

def evaluate_baseline(test_set,
            baseline_model, 
            epochs=100, 
            shot=1,
            way=5,
            querysize=32, 
            verbose=0, 
            load_weights_dirname=None,
            save_finetune_dirname=None
            ):
    test_set = loader.load_miniImgnet(test_path)
    # create dataset for training, validation, testing
    train_img, train_label, val_img, val_label, test_img, test_label, labels_pick  = \
        tester.create_finetuningset(dataset=test_set, way=way, shot=shot, querysize=querysize)
    train_img = train_img/255.
    val_img = val_img/255.
    test_img = test_img/255.


    input_shape = train_img.shape[1:]

    # baseline
    # baseline = Baseline(input_shape=input_shape,
    #                     classes=way,
    #                     loadbb=True,
    #                     load_weigths_path=os.path.join("../weights/{}/{}".format(load_weights_dirname, \
    #                                 'baseline_training.h5')),
    #                     freezebb=True)
    # baseline.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=1e-3),
    #               metrics=['accuracy'])
    baseline_model.load_weights(os.path.join("../weights/{}/{}".format(load_weights_dirname, \
                                    'baseline_training.h5')), by_name=True, skip_mismatch=True)
    print(K.eval(baseline_model.layers[-1].bias))

    history_baseline = trainer.train_model(model=baseline_model,
                    x=train_img, y=train_label,
                    shuffle=True,
                    val_data=(val_img, val_label),
                    batch_size=querysize,
                    epochs=epochs ,
                    save_weights_path=os.path.join("../weights/{}/{}".format(save_finetune_dirname, \
                        'baseline_finetuning_{}shot.h5'.format(shot))),
                    monitor='val_acc',
                    verbose=verbose
                    )
   
    # load best model and 
    baseline_model.load_weights('../weights/{}/baseline_finetuning_{}shot.h5'.format(save_finetune_dirname, shot))
    performance_baseline_finetune = baseline_model.evaluate(x=test_img, y=test_label)

    dirpath = "../results/{}".format(save_finetune_dirname)
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    with open(os.path.join(dirpath, "result_baseline.txt"), "a") as file:
        np.savetxt(file, np.array(performance_baseline_finetune).reshape(1, -1))
    
if __name__ == "__main__":
    print("Start")
    from time import time

    test_path = os.path.join("../data/MiniImagenet", 'miniImageNet_category_split_test.pickle')

    EXP_NAME = "Exp_CUB_100" #"Exp_200" ## Exp_CUB_101
    EXP_NAME_fine_tune = "Exp_FineTune_CUB_100_5_shot"##"Exp_FineTune_200"
    

    shot = 5
    finetune_classes = 5
    finetune_epoch = 100


    # baseline = Baseline(input_shape=(84, 84, 3),
    #                     classes=finetune_classes,
    #                     loadbb=True,
    #                     load_weigths_path=os.path.join("../weights/{}/{}".format(EXP_NAME, \
    #                                 'baseline_training.h5')),
    #                     freezebb=True)
    # baseline.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=1e-3),
    #               metrics=['accuracy'])

    # baseline.summary()
    # import ipdb; ipdb.set_trace()
    st = time()
    current_st = st
    for i in range(300):
        if i%10 == 0:
            current_et = time()
            print("="*50)
            print("Iter {}, time elapsed {:.2f}".format(i, current_et-current_st))
            print("="*50)
            current_st = current_et
        evaluate(test_path,
            epochs=finetune_epoch,
            querysize=32,
            shot=shot,
            way=finetune_classes,
            load_weights_dirname=EXP_NAME,
            save_finetune_dirname=EXP_NAME_fine_tune)
        # evaluate_baseline(test_path,
        #     baseline,
        #     epochs=finetune_epoch,
        #     querysize=32,
        #     shot=shot,
        #     way=finetune_classes,
        #     load_weights_dirname=EXP_NAME,
        #     save_finetune_dirname=EXP_NAME_fine_tune)
        K.clear_session()

    
    print("Total elapsed time {:.2f}".format(time()-st))
