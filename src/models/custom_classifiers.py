from keras import backend
# from src.models.custom_layers import CosDense
from custom_layers import CosDense
from keras import layers
from keras import models


def Cos_classifier(input_shape,
                       #feature_extracted,
                       classes,
                       activation='softmax'):
    """
    :param feature_extraxted: the input tensor
    :param n_classes:
    :return: the classifier model
    """
    input_tensor = layers.Input(shape=input_shape)
    # x = CosDense(classes,activation=activation, name='cosine_dense')(feature_extracted)
    x = CosDense(classes, activation=activation, name='cosine_dense1')(input_tensor)
    model = models.Model(input_tensor, x, name='cos_classifier')
    return model
