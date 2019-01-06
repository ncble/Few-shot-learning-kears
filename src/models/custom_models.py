from keras import backend
from keras import layers
from keras import models
from src.models.backbones import ResNet10
from src.models.custom_classifiers import Cos_classifier
from keras import optimizers
import os

base_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..','..')
weights_folder_path = os.path.join(base_path,'weights')


def Baseline_plus(input_shape=None,classes=None,loadbb=False,weights_file=None,freezebb=False):
    """

    :param input_shape:
    :param classes:
    :param loadbb:
    :param weights_pth: here,we load weights by name,
                        so we can use the whole pretrained baseline++ weights
    :param freezebb:
    :return:
    """
    # input tensor
    # img_input = layers.Input(shape=input_shape)
    # backbone
    backbone = ResNet10(input_shape=input_shape,
             include_top=False,
             pooling='avg',
             classes=None)
    if loadbb:
        weights_path = os.path.join(weights_folder_path,weights_file)
        backbone.load_weights(weights_path,by_name=True)
    if freezebb:
        for layer in backbone.layers:
            layer.trainable = False

    #classifier based on cosine similarity
    classifier = Cos_classifier(input_shape=backbone.output_shape[1:],classes=classes)
    # inputs = img_input
    # model = models.Model(inputs, x, name='baseline_plus')
    model = models.Model(inputs=backbone.input,outputs=classifier(backbone.output))
    # model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizers.Adam(lr=1e-3),
    #                       metrics=['accuracy'])
    return model

def Baseline(input_shape=None,classes=None,loadbb=False,weights_file=None,freezebb=False):
    """

    :param input_shape:
    :param classes:
    :param loadbb:
    :param weights_pth: here,we load weights by name,
                        so we can use the whole pretrained baseline++ weights
    :param freezebb:
    :return:
    """
    # input tensor
    # img_input = layers.Input(shape=input_shape)
    # backbone
    backbone = ResNet10(input_shape=input_shape,
             include_top=False,
             pooling='avg',
             classes=None)
    if loadbb:
        weights_path = os.path.join(weights_folder_path,weights_file)
        backbone.load_weights(weights_path,by_name=True)
    if freezebb:
        for layer in backbone.layers:
            layer.trainable = False

    #classifier based on cosine similarity
    #classifier = Cos_classifier(input_shape=backbone.output_shape[1:],classes=classes)

    # inputs = img_input
    # model = models.Model(inputs, x, name='baseline_plus')
    model = models.Model(inputs=backbone.input,outputs=layers.Dense(classes, activation='softmax', name='fc_final')(backbone.output))
    # model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizers.Adam(lr=1e-3),
    #                       metrics=['accuracy'])
    return model

