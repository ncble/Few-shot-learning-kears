from src.utils import trainer,loader,tester,saver
from src.models.custom_models import Baseline_plus,Baseline
#from src.models.backbones import ResNet10
from keras import optimizers
import pickle


####################### training stage ###################
# The training stage is for training backbone
#########################################################
# # the index for the experiment of training stage
# training_experiment = 'training_setting'
#
# # load image
# train_train_path = 'miniImageNet_category_split_train_phase_train.pickle'
# train_val_path = 'miniImageNet_category_split_train_phase_val.pickle'
#
# train_train = loader.load_miniImgnet(train_train_path)
# train_val = loader.load_miniImgnet(train_val_path)
#
# train_img = train_train[b'data']
# train_label = train_train[b'labels']
#
# val_img = train_val[b'data']
# val_label = train_val[b'labels']
#
# # define input shape, classes
# input_shape = train_img.shape[1:]
# classes = len(train_train[b'catname2label'])
# train_epoch = 400
#
# # baseline
# baseline = Baseline(input_shape=input_shape,classes=classes)
# baseline.compile(loss='sparse_categorical_crossentropy', optimizer=optimizers.Adam(lr=1e-3),
#               metrics=['accuracy'])
#
# history_baseline = trainer.train_model(model=baseline,
#                 x=train_img,y=train_label,
#                 shuffle=True,
#                 val_data=(val_img,val_label),
#                 batch_size=64,
#                 epochs=train_epoch ,
#                 foldername=training_experiment,
#                 weights_file='baseline_training.h5', # save weights
#                 monitor='val_acc',
#                 )
#
# saver.save_output(history_baseline,training_experiment,'history_baseline_training.pickle')
#
# # baseline_plus
# baseline_plus = Baseline_plus(input_shape=input_shape,classes=classes)
# baseline_plus.compile(loss='sparse_categorical_crossentropy',
#                       optimizer=optimizers.Adam(lr=1e-3),
#                       metrics=['accuracy'])
# history_baseline_plus = trainer.train_model(model=baseline_plus,
#                 x=train_img,y=train_label,
#                 shuffle=True,
#                 val_data=(val_img,val_label),
#                 batch_size=64,
#                 epochs=train_epoch ,
#                 foldername=training_experiment,
#                 weights_file='baseline_plus_training.h5', # save weights
#                 monitor='val_acc',
#                 )
# saver.save_output(history_baseline_plus,training_experiment,'history_baseline_plus_training.pickle')
#
#
# # release the variables
# del train_train, train_val, train_img, train_label, val_img, val_label
# del baseline, baseline_plus, history_baseline, history_baseline_plus
#

# ##################### fine-tuning stage ######################
# for training a new classifier for novel classes
################################################################
# the index for the experiment of finetuning stage
finetuning_experiment = 'finetuning_setting'
loadbb_experiment= 'training_setting'


test_path = 'miniImageNet_category_split_test.pickle'
test = loader.load_miniImgnet(test_path)


classes=5
shot = 1
finetune_epoch = 100

# create dataset for training, validation, testing
train_img,train_label,val_img,val_label,test_img,test_label,labels_pick  = \
    tester.create_finetuningset(dataset=test,way=classes,shot=shot,querysize=16)

saver.save_output(labels_pick,finetuning_experiment,'labelspick_{}shot.pickle'.format(shot))



input_shape = train_img.shape[1:]


# baseline
baseline = Baseline(input_shape=input_shape,
                    classes=classes,
                    loadbb=True,
                    folder_weights=loadbb_experiment,
                    weights_file='baseline_training.h5', # load weights
                    freezebb=True)
baseline.compile(loss='sparse_categorical_crossentropy', optimizer=optimizers.Adam(lr=1e-3),
              metrics=['accuracy'])

history_baseline = trainer.train_model(model=baseline,
                x=train_img,y=train_label,
                shuffle=True,
                val_data=(val_img,val_label),
                batch_size=16,
                epochs=finetune_epoch ,
                foldername=finetuning_experiment,
                weights_file='baseline_finetuning_{}shot.h5'.format(shot),# save weights
                monitor='val_acc',
                )

saver.save_output(history_baseline,finetuning_experiment,'history_baseline_finetuning_{}shot.pickle'.format(shot))


# load best model and evaluate
baseline.load_weights('weights/{}/baseline_finetuning_{}shot.h5'.format(finetuning_experiment,shot))
performance_baseline_finetune = baseline.evaluate(x=test_img,y=test_label)
saver.save_output(performance_baseline_finetune,finetuning_experiment,'performance_baseline_finetune_{}shot.pickle'.format(shot))


# baseline plus
baseline_plus = Baseline_plus(input_shape=input_shape,
                    classes=classes,
                    loadbb=True,
                    folder_weights=loadbb_experiment,
                    weights_file='baseline_plus_training.h5', # load weights
                    freezebb=True)
baseline_plus.compile(loss='sparse_categorical_crossentropy', optimizer=optimizers.Adam(lr=1e-3),
              metrics=['accuracy'])

history_baseline_plus = trainer.train_model(model=baseline_plus,
                x=train_img,y=train_label,
                shuffle=True,
                val_data=(val_img,val_label),
                batch_size=16,
                epochs=finetune_epoch ,
                foldername=finetuning_experiment,
                weights_file='baseline_plus_finetuning_{}shot.h5'.format(shot),# save weights
                monitor='val_acc',
                )
saver.save_output(history_baseline_plus,finetuning_experiment,'history_baseline_plus_finetuning_{}shot.pickle'.format(shot))

# load best model and evaluate
baseline_plus.load_weights('weights/{}/baseline_plus_finetuning_{}shot.h5'.format(finetuning_experiment,shot))
performance_baseline_plus_finetune = baseline_plus.evaluate(x=test_img,y=test_label)
saver.save_output(performance_baseline_plus_finetune,finetuning_experiment,'performance_baseline_plus_finetune_{}shot.pickle'.format(shot))
