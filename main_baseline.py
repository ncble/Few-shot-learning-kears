from src.utils import trainer,loader,tester,saver
from src.models.custom_models import Baseline_plus,Baseline
#from src.models.backbones import ResNet10
from keras import optimizers
import pickle


####################### training stage ###################
# The training stage is for training backbone
#########################################################
# # the index for the experiment of training stage
# training_experiment = 0
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
#
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
#                 epochs=400 ,
#                 weights_file='exp{}_baseline_training.h5'.format(training_experiment), # save weights
#                 monitor='val_acc',
#                 )
#
# saver.save_output(history_baseline,'exp{}_history_baseline_training.pickle'.format(training_experiment))
# #pickle.dump(history_baseline,open('output/exp{}_history_baseline_training.pickle'.format(training_experiment),'wb'))
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
#                 epochs=400 ,
#                 weights_file='exp{}_baseline_plus_training.h5'.format(training_experiment), # save weights
#                 monitor='val_acc',
#                 )
# saver.save_output(history_baseline_plus,'exp{}_history_baseline_plus_training.pickle'.format(training_experiment))
# #pickle.dump(history_baseline_plus,open('output/exp{}_history_baseline_plus_training.pickle'.format(training_experiment),'wb'))
#
#
# # release the variables
# del train_train, train_val, train_img, train_label, val_img, val_label
# del baseline, baseline_plus, history_baseline, history_baseline_plus


# ##################### fine-tuning stage ######################
# for training a new classifier for novel classes
################################################################
# the index for the experiment of finetuning stage
finetuning_experiment = 0
# denotes the backbone of which training experiment will be used
loadbb_experiment = 0


test_path = 'miniImageNet_category_split_test.pickle'
test = loader.load_miniImgnet(test_path)


classes=5
shot = 1

# create dataset for training, validation, testing
train_img,train_label,val_img,val_label,test_img,test_label,labels_pick  = \
    tester.create_finetuningset(dataset=test,way=classes,shot=shot,querysize=16)

saver.save_output(labels_pick,'exp{}_load{}_labelspick_{}shot.pickle'.format(finetuning_experiment,loadbb_experiment,shot))
#pickle.dump(labels_pick,open('output/exp{}_load{}_labelspick_{}shot.pickle'.format(finetuning_experiment,loadbb_experiment,shot),'wb'))


input_shape = train_img.shape[1:]


# baseline
baseline = Baseline(input_shape=input_shape,
                    classes=classes,
                    loadbb=True,
                    weights_file='exp{}_baseline_training.h5'.format(loadbb_experiment), # load weights
                    freezebb=True)
baseline.compile(loss='sparse_categorical_crossentropy', optimizer=optimizers.Adam(lr=1e-3),
              metrics=['accuracy'])

history_baseline = trainer.train_model(model=baseline,
                x=train_img,y=train_label,
                shuffle=True,
                val_data=(val_img,val_label),
                batch_size=16,
                epochs=100 ,
                weights_file='exp{}_load{}_baseline_finetuning_{}shot.h5'.format(finetuning_experiment,loadbb_experiment,shot),# save weights
                monitor='val_acc',
                )

saver.save_output(history_baseline,'exp{}_load{}_history_baseline_finetuning_{}shot.pickle'.format(finetuning_experiment,loadbb_experiment,shot))
#pickle.dump(history_baseline,open('output/exp{}_load{}_history_baseline_finetuning_{}shot.pickle'.format(finetuning_experiment,loadbb_experiment,shot),'wb'))
# load best model and evaluate
baseline.load_weights('weights/exp{}_load{}_baseline_finetuning_{}shot.h5'.format(finetuning_experiment,loadbb_experiment,shot))
performance_baseline_finetune = baseline.evaluate(x=test_img,y=test_label)
saver.save_output(performance_baseline_finetune,'exp{}_load{}_performance_baseline_finetune_{}shot.pickle'.format(finetuning_experiment,loadbb_experiment,shot))
#pickle.dump(performance_baseline_finetune,open('output/exp{}_load{}_performance_baseline_finetune_{}shot.pickle'.format(finetuning_experiment,loadbb_experiment,shot),'wb'))


# baseline plus
baseline_plus = Baseline_plus(input_shape=input_shape,
                    classes=classes,
                    loadbb=True,
                    weights_file='exp{}_baseline_plus_training.h5'.format(loadbb_experiment), # load weights
                    freezebb=True)
baseline_plus.compile(loss='sparse_categorical_crossentropy', optimizer=optimizers.Adam(lr=1e-3),
              metrics=['accuracy'])

history_baseline_plus = trainer.train_model(model=baseline_plus,
                x=train_img,y=train_label,
                shuffle=True,
                val_data=(val_img,val_label),
                batch_size=16,
                epochs=100 ,
                weights_file='exp{}_load{}_baseline_plus_finetuning_{}shot.h5'.format(finetuning_experiment,loadbb_experiment,shot),# save weights
                monitor='val_acc',
                )
saver.save_output(history_baseline_plus,'exp{}_load{}_history_baseline_plus_finetuning_{}shot.pickle'.format(finetuning_experiment,loadbb_experiment,shot))
#pickle.dump(history_baseline_plus,open('output/exp{}_load{}_history_baseline_plus_finetuning_{}shot.pickle'.format(finetuning_experiment,loadbb_experiment,shot),'wb'))
# load best model and evaluate
baseline_plus.load_weights('weights/exp{}_load{}_baseline_plus_finetuning_{}shot.h5'.format(finetuning_experiment,loadbb_experiment,shot))
performance_baseline_plus_finetune = baseline_plus.evaluate(x=test_img,y=test_label)
saver.save_output(performance_baseline_plus_finetune,'exp{}_load{}_performance_baseline_plus_finetune_{}shot.pickle'.format(finetuning_experiment,loadbb_experiment,shot))
#pickle.dump(performance_baseline_plus_finetune,open('output/exp{}_load{}_performance_baseline_plus_finetune_{}shot.pickle'.format(finetuning_experiment,loadbb_experiment,shot),'wb'))