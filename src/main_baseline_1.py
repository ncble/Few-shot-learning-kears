import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
gpu_fraction = 0.95
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
EXP_NAME = "Exp_debug0"
EXP_NAME_fine_tune = "Exp_debug0_fine_tuning"
train_epoch = 1

classes = 64 ## useless TODO
shot = 1
finetune_classes = 5
finetune_epoch = 100

finetune_dirname = EXP_NAME_fine_tune #'finetuning_setting'
loadbb_experiment = EXP_NAME#'training_setting'

train_train_path = os.path.join("../data/MiniImagenet", 'miniImageNet_category_split_train_phase_train.pickle')
train_val_path = os.path.join("../data/MiniImagenet", 'miniImageNet_category_split_train_phase_val.pickle')
test_path = os.path.join("../data/MiniImagenet", 'miniImageNet_category_split_test.pickle')


#######################

# load image
# train_train_path = 'miniImageNet_category_split_train_phase_train.pickle'
# train_val_path = 'miniImageNet_category_split_train_phase_val.pickle'

train_train = loader.load_miniImgnet(train_train_path)
train_val = loader.load_miniImgnet(train_val_path)

train_img = train_train[b'data']
train_label = train_train[b'labels']

val_img = train_val[b'data']
val_label = train_val[b'labels']

# define input shape, classes
input_shape = train_img.shape[1:]
classes = len(train_train[b'catname2label'])



# baseline
# baseline = Baseline(input_shape=input_shape, classes=classes)
# baseline.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=1e-3), 
#               metrics=['accuracy'])

# history_baseline = trainer.train_model(model=baseline,
#                     x=train_img, y=train_label,
#                     shuffle=True,
#                     val_data=(val_img, val_label),
#                     batch_size=64,
#                     epochs=train_epoch ,
#                     # foldername=EXP_NAME,
#                     # weights_file='baseline_training.h5', # save weights
#                     save_weights_path=os.path.join("../weights/{}/{}".format(EXP_NAME, \
#                                         'baseline_training.h5')),
#                     monitor='val_acc',
#                     )
# out_filepath = os.path.join("../output/{}/{}".format(EXP_NAME, 'history_baseline_training.pickle'))
# saver.save_output(history_baseline, out_filepath)

# # baseline_plus
# baseline_plus = Baseline_plus(input_shape=input_shape, classes=classes)
# baseline_plus.compile(loss='sparse_categorical_crossentropy',
#                       optimizer=Adam(lr=1e-3),
#                       metrics=['accuracy'])
# history_baseline_plus = trainer.train_model(model=baseline_plus,
#                 x=train_img, y=train_label,
#                 shuffle=True,
#                 val_data=(val_img, val_label),
#                 batch_size=64,
#                 epochs=train_epoch ,
#                 # foldername=EXP_NAME,
#                 # weights_file='baseline_plus_training.h5', # save weights
#                 save_weights_path=os.path.join("../weights/{}/{}".format(EXP_NAME, \
#                                         'baseline_plus_training.h5')),
#                 monitor='val_acc',
#                 )
# out_filepath = os.path.join("../output/{}/{}".format(EXP_NAME, 'history_baseline_plus_training.pickle'))
# saver.save_output(history_baseline_plus, out_filepath)

# # release the variables
# del train_train, train_val, train_img, train_label, val_img, val_label
# del baseline, baseline_plus, history_baseline, history_baseline_plus



# ##################### fine-tuning stage ######################
# for training a new classifier for novel classes
################################################################
# the weights will be saved at 'weights/finetuning_setting/'
# the output will be saved at 'output/finetuning_setting/'
# if 'finetuning_setting' does not exist under 'weights/' or 'output/', it will be created automatically
# finetune_dirname = EXP_NAME_fine_tune #'finetuning_setting'
# loadbb_experiment = EXP_NAME#'training_setting'


# test_path = 'miniImageNet_category_split_test.pickle'
test_set = loader.load_miniImgnet(test_path)


# finetune_classes=5
# shot = 1
# finetune_epoch = 100

# create dataset for training, validation, testing
train_img, train_label, val_img, val_label, test_img, test_label, labels_pick  = \
    tester.create_finetuningset(dataset=test_set, way=finetune_classes, shot=shot, querysize=16)


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
baseline.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=1e-3),
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
baseline_plus.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=1e-3),
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
