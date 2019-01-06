from src.utils import trainer
from src.models.custom_models import Baseline_plus,Baseline
from src.utils import loader
from src.models.backbones import ResNet10
from keras import optimizers
import pickle
from src.utils import tester

####################### training stage ###################
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
# #print(input_shape)
# classes = len(train_train[b'catname2label'])
# #print(classes)
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
#                 weights_file='baseline_training.h5',
#                 monitor='val_loss',
#                 )
#
# pickle.dump(history_baseline,open('output/history_baseline.pickle','wb'))
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
#                 weights_file='baseline_plus_training.h5',
#                 monitor='val_loss',
#                 )
# pickle.dump(history_baseline_plus,open('output/history_baseline_plus.pickle','wb'))
#

##################### fine-tuning stage ######################

test_path = 'miniImageNet_category_split_test.pickle'
test = loader.load_miniImgnet(test_path)

# create dataset for training, validation, testing
classes=5
train_img,train_label,val_img,val_label,test_img,test_label = \
    tester.create_finetuningset(dataset=test,way=classes,shot=1,querysize=16)

input_shape = train_img.shape[1:]

# baseline
baseline = Baseline(input_shape=input_shape,
                    classes=classes,
                    loadbb=True,
                    weights_file='baseline_training.h5', # load weights
                    freezebb=True)
baseline.compile(loss='sparse_categorical_crossentropy', optimizer=optimizers.Adam(lr=1e-3),
              metrics=['accuracy'])

history_baseline = trainer.train_model(model=baseline,
                x=train_img,y=train_label,
                shuffle=True,
                val_data=(val_img,val_label),
                batch_size=16,
                epochs=100 ,
                weights_file='baseline_finetuning.h5',# save weights
                monitor='val_loss',
                )
pickle.dump(history_baseline,open('output/history_baseline_finetuning.pickle','wb'))
# load best model and evaluate
baseline.load_weights('weights/baseline_finetuning.h5')
performance_baseline_finetune = baseline.evaluate(x=test_img,y=test_label)
pickle.dump(performance_baseline_finetune,open('output/performance_baseline_finetune','wb'))

# baseline plus
baseline_plus = Baseline_plus(input_shape=input_shape,
                    classes=classes,
                    loadbb=True,
                    weights_file='baseline_plus_training.h5', # load weights
                    freezebb=True)
baseline_plus.compile(loss='sparse_categorical_crossentropy', optimizer=optimizers.Adam(lr=1e-3),
              metrics=['accuracy'])

history_baseline_plus = trainer.train_model(model=baseline_plus,
                x=train_img,y=train_label,
                shuffle=True,
                val_data=(val_img,val_label),
                batch_size=16,
                epochs=100 ,
                weights_file='baseline_plus_finetuning.h5',# save weights
                monitor='val_loss',
                )
pickle.dump(history_baseline_plus,open('output/history_baseline_plus_finetuning.pickle','wb'))
# load best model and evaluate
baseline_plus.load_weights('weights/baseline_plus_finetuning.h5')
performance_baseline_plus_finetune = baseline_plus.evaluate(x=test_img,y=test_label)
pickle.dump(performance_baseline_plus_finetune,open('output/performance_baseline_plus_finetune','wb'))