from src.models.backbones import ResNet10
from keras import backend as K
import tensorflow as tf
from src.models.custom_models import Baseline_plus,Baseline
import pickle
import matplotlib.pyplot as plt
from keras import optimizers
from src.utils import loader

##################################
# test backbone
##################################
# model1 = ResNet10(include_top=True,
#              input_shape=(224,224,3),
#              pooling=None,
#              classes=10)
# print(model1.summary())
#
#
# model2 = ResNet10(include_top=False,
#              input_shape=(224,224,3),
#              pooling=True,
#              classes=10)
# print(model2.summary())


#######################################
# # check cos_dense layer
# def l2_norm(x, axis=None):
#     """
#     takes an input tensor and returns the l2 norm along specified axis
#     """
#     square_sum = K.sum(K.square(x), axis=axis, keepdims=True)
#     norm = K.sqrt(K.maximum(square_sum, K.epsilon()))
#
#     return norm
#
# x = K.placeholder(shape=(15, 4))
# y = K.placeholder(shape=(4, 2))
# xy = K.dot(x,y)
# x_norm = l2_norm(x,axis=-1)
# y_norm = l2_norm(y,axis=0)
# xy_norm = K.dot(x_norm,y_norm)
#
# cos_similarity = xy/xy_norm
# print(cos_similarity)
#######################################
# check baseline_plus,baseline
#
# train_val_path = 'miniImageNet_category_split_train_phase_val.pickle'
# train_val = loader.load_miniImgnet(train_val_path)
# val_img = train_val[b'data']
# val_label = train_val[b'labels']
#
# # define input shape, classes
# input_shape = val_img.shape[1:]
# classes = len(train_val[b'catname2label'])

# baseline = Baseline(input_shape=input_shape,
#                     classes=classes)
# baseline.compile(loss='sparse_categorical_crossentropy', optimizer=optimizers.Adam(lr=1e-3),
#               metrics=['accuracy'])
# baseline.load_weights('weights/baseline_training.h5')
# performance = baseline.evaluate(x=val_img,y=val_label)

# baseline_plus = Baseline_plus(input_shape=input_shape,
#                     classes=classes)
# baseline_plus.compile(loss='sparse_categorical_crossentropy', optimizer=optimizers.Adam(lr=1e-3),
#               metrics=['accuracy'])
# baseline_plus.load_weights('weights/baseline_plus_training.h5')
# performance = baseline_plus.evaluate(x=val_img,y=val_label)
#
#
# print(performance)
#print(model.summary())

#######################################
# check baseline training history
# history_baseline = pickle.load(open('output/history_baseline.pickle','rb'))
# plt.plot(history_baseline.history['acc'])
#plt.plot(history_baseline.history['loss'])
#plt.plot(history_baseline.history['val_loss'])
# plt.plot(history_baseline.history['val_acc'])
# plt.show()

# check finetuning performance
per_baseline = pickle.load(open('output/performance_baseline_finetune_1shot.pickle','rb'))
print(per_baseline)
per_baseline_plus = pickle.load(open('output/performance_baseline_plus_finetune_1shot.pickle','rb'))
print(per_baseline_plus)


per_baseline = pickle.load(open('output/performance_baseline_finetune_5shot.pickle','rb'))
print(per_baseline)
per_baseline_plus = pickle.load(open('output/performance_baseline_plus_finetune_5shot.pickle','rb'))
print(per_baseline_plus)
