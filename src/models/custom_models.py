# from keras import backend
from keras.layers import Dense
from keras.models import Model
# from src.models.backbones import ResNet10
from models.backbones import ResNet10
from models.custom_classifiers import Cos_classifier
# from custom_classifiers import Cos_classifier
from keras import optimizers
import os

# base_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..')
# weights_folder_path = os.path.join(base_path, 'weights')


def Baseline_plus(input_shape=None, 
                    classes=None,
                    loadbb=False, 
                    # folder_weights=None,
                    # weights_file=None,
                    load_weigths_path=None,
                    freezebb=False):
    """

    :param input_shape:
    :param classes:
    :param loadbb: if load weights to backbone
    :param weights_pth: here,we load weights by name, so that only the weights of backbone is loaded
    :param freezebb: if freeze the weights of backbone
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
        assert load_weigths_path is not None, "weights path is None"
        # weights_path = os.path.join(weights_folder_path, folder_weights, weights_file)

        backbone.load_weights(load_weigths_path, by_name=True)
    if freezebb:
        for layer in backbone.layers:
            layer.trainable = False

    #classifier based on cosine similarity
    classifier = Cos_classifier(input_shape=backbone.output_shape[1:], classes=classes)
    # inputs = img_input
    # model = models.Model(inputs, x, name='baseline_plus')
    model = Model(inputs=backbone.input, outputs=classifier(backbone.output))
    # model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizers.Adam(lr=1e-3),
    #                       metrics=['accuracy'])
    return model

def Baseline(input_shape=None, 
            classes=None, 
            loadbb=False, 
            # folder_weights=None, 
            # weights_file=None, 
            load_weigths_path=None,
            freezebb=False):
    """

    :param input_shape:
    :param classes:
    :param loadbb: if load weights to backbone
    :param weights_pth: here,we load weights by name, so that only the weights of backbone is loaded
    :param freezebb: if freeze the weights of backbone
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
        assert load_weigths_path is not None, "weights path is None"
        # weights_path = os.path.join(weights_folder_path, folder_weights, weights_file)
        backbone.load_weights(load_weigths_path, by_name=True)
    if freezebb:
        for layer in backbone.layers:
            layer.trainable = False

    #classifier based on cosine similarity
    #classifier = Cos_classifier(input_shape=backbone.output_shape[1:],classes=classes)

    # inputs = img_input
    # model = models.Model(inputs, x, name='baseline_plus')
    model = Model(inputs=backbone.input, outputs=Dense(classes, activation='softmax', name='fc_final')(backbone.output))
    # model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizers.Adam(lr=1e-3),
    #                       metrics=['accuracy'])
    return model

