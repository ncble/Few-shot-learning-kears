import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--gpu_num", help="Choose the number of GPU.", type=str, default="0", choices=["0", "1"])
parser.add_argument("--gpu_fra", help="Decide the GPU fraction to use.", type=float, default=0.95)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_num
gpu_fraction = args.gpu_fra

import numpy as np
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


####################### training stage ###################
# The training stage is for training backbone
#########################################################
# the weights will be saved at 'weights/training_setting/'
# the output will be saved at 'output/training_setting/'
# if 'training_setting' does not exist under 'weights/' or 'output/', it will be created automatically


#######################
##      Config
#######################
EXP_NAME = "Exp_CUB_200"
EXP_NAME_fine_tune = "Exp_CUB_200_fine_tuning"
train_epoch = 300

classes = 100 #
shot = 5
finetune_classes = 5 ## way
finetune_epoch = 100
optimizer_lr = 2*1e-3
finetune_dirname = EXP_NAME_fine_tune #'finetuning_setting'
loadbb_experiment = EXP_NAME#'training_setting'

# train_train_path = os.path.join("../data/MiniImagenet", 'miniImageNet_category_split_train_phase_train.pickle')
# train_val_path = os.path.join("../data/MiniImagenet", 'miniImageNet_category_split_train_phase_val.pickle')
# test_path = os.path.join("../data/MiniImagenet", 'miniImageNet_category_split_test.pickle')
train_trainX_path = os.path.join("../data/CUB_128/CUB_meta_train_train_X.npy")
train_trainY_path = os.path.join("../data/CUB_128/CUB_meta_train_train_Y.npy")
train_valX_path = os.path.join("../data/CUB_128/CUB_meta_train_valid_X.npy")
train_valY_path = os.path.join("../data/CUB_128/CUB_meta_train_valid_Y.npy")

test_X_path = os.path.join("../data/CUB_128/CUB_meta_test_X.npy")
test_Y_path = os.path.join("../data/CUB_128/CUB_meta_test_Y.npy")



#######################

# load image
# train_train_path = 'miniImageNet_category_split_train_phase_train.pickle'
# train_val_path = 'miniImageNet_category_split_train_phase_val.pickle'

# train_train = loader.load_miniImgnet(train_train_path) 
# train_val = loader.load_miniImgnet(train_val_path)

train_img = np.load(train_trainX_path)
train_label = np.load(train_trainY_path) 

val_img = np.load(train_valX_path)
val_label = np.load(train_valY_path)


train_img = train_img/255
val_img = val_img/255

# define input shape, classes
input_shape = train_img.shape[1:]




# baseline
baseline = Baseline(input_shape=input_shape, classes=classes)
baseline.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=optimizer_lr), 
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
                      optimizer=Adam(lr=optimizer_lr),
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



del train_img, train_label, val_img, val_label
del baseline, baseline_plus, history_baseline, history_baseline_plus



# ##################### fine-tuning stage ######################
# for training a new classifier for novel classes
################################################################
# the weights will be saved at 'weights/finetuning_setting/'
# the output will be saved at 'output/finetuning_setting/'
# if 'finetuning_setting' does not exist under 'weights/' or 'output/', it will be created automatically
# finetune_dirname = EXP_NAME_fine_tune #'finetuning_setting'
# loadbb_experiment = EXP_NAME#'training_setting'


# test_path = 'miniImageNet_category_split_test.pickle'
# test_set = loader.load_miniImgnet(test_path)


test_X = np.load(test_X_path)
test_Y = np.load(test_Y_path)


# finetune_classes=5
# shot = 1
# finetune_epoch = 100

# create dataset for training, validation, testing
train_img, train_label, val_img, val_label, test_img, test_label, labels_pick  = \
    loader.load_fewshot_testset(test_X, test_Y, way=finetune_classes, shot=shot, querysize=16)

train_img = train_img/255.
val_img = val_img/255.
test_img = test_img/255.


out_filepath = os.path.join("../output/{}/{}".format(finetune_dirname, 'labelspick_{}shot.pickle'.format(shot)))
saver.save_output(labels_pick, out_filepath)



input_shape = train_img.shape[1:]

# import ipdb; ipdb.set_trace()
# baseline
baseline = Baseline(input_shape=input_shape,
                    classes=finetune_classes,
                    loadbb=True,
                    # folder_weights=loadbb_experiment,
                    # weights_file='baseline_training.h5', # load weights
                    load_weigths_path=os.path.join("../weights/{}/{}".format(loadbb_experiment, \
                                'baseline_training.h5')),
                    freezebb=True)
baseline.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=optimizer_lr),
              metrics=['accuracy'])

history_baseline = trainer.train_model(model=baseline,
                x=train_img, y=train_label,
                shuffle=True,
                val_data=(val_img, val_label),
                batch_size=16,
                epochs=finetune_epoch ,
                # foldername=finetune_dirname,
                # weights_file='baseline_finetuning_{}shot.h5'.format(shot),# save weights
                save_weights_path=os.path.join("../weights/{}/{}".format(finetune_dirname, \
                    'baseline_finetuning_{}shot.h5'.format(shot))),
                monitor='val_acc',
                )
out_filepath = os.path.join("../output/{}/{}".format(finetune_dirname, 'history_baseline_finetuning_{}shot.pickle'.format(shot)))
saver.save_output(history_baseline, out_filepath)

# import ipdb; ipdb.set_trace()
# load best model and evaluate
baseline.load_weights('../weights/{}/baseline_finetuning_{}shot.h5'.format(finetune_dirname, shot))
performance_baseline_finetune = baseline.evaluate(x=test_img, y=test_label)

print(performance_baseline_finetune)

out_filepath = os.path.join("../output/{}/{}".format(finetune_dirname, 'performance_baseline_finetune_{}shot.pickle'.format(shot)))
saver.save_output(performance_baseline_finetune, out_filepath)


# baseline plus
baseline_plus = Baseline_plus(input_shape=input_shape,
                    classes=finetune_classes,
                    loadbb=True,
                    # folder_weights=loadbb_experiment,
                    # weights_file='baseline_plus_training.h5', # load weights
                    load_weigths_path=os.path.join("../weights/{}/{}".format(loadbb_experiment, \
                                'baseline_plus_training.h5')),
                    freezebb=True)
baseline_plus.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=optimizer_lr),
              metrics=['accuracy'])

history_baseline_plus = trainer.train_model(model=baseline_plus,
                x=train_img, y=train_label,
                shuffle=True,
                val_data=(val_img, val_label),
                batch_size=16,
                epochs=finetune_epoch,
                save_weights_path=os.path.join("../weights/{}/{}".format(finetune_dirname, \
                    'baseline_plus_finetuning_{}shot.h5'.format(shot))),
                # foldername=finetune_dirname,
                # weights_file='baseline_plus_finetuning_{}shot.h5'.format(shot),# save weights
                monitor='val_acc',
                )
out_filepath = os.path.join("../output/{}/{}".format(finetune_dirname, 'history_baseline_plus_finetuning_{}shot.pickle'.format(shot)))
saver.save_output(history_baseline_plus, out_filepath)

# load best model and evaluate
baseline_plus.load_weights('../weights/{}/baseline_plus_finetuning_{}shot.h5'.format(finetune_dirname, shot))
performance_baseline_plus_finetune = baseline_plus.evaluate(x=test_img, y=test_label)

out_filepath = os.path.join("../output/{}/{}".format(finetune_dirname, 'performance_baseline_plus_finetune_{}shot.pickle'.format(shot)))
saver.save_output(performance_baseline_plus_finetune, out_filepath)
print(performance_baseline_plus_finetune)