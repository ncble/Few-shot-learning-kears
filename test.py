from src.models.backbones import ResNet10
from keras import backend as K
import tensorflow as tf
from src.models.custom_models import Baseline_plus
import pickle

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
# check baseline_plus
model = Baseline_plus(input_shape=(84,84,3),classes=10)
print(model.summary())