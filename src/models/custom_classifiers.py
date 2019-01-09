from keras import backend
from models.custom_layers import CosDense
# from custom_layers import CosDense
from keras.layers import Input
from keras.models import Model


def Cos_classifier(input_shape,
                       #feature_extracted,
                       classes,
                       activation='softmax'):
    """
    :param feature_extraxted: the input tensor
    :param n_classes:
    :return: the classifier model
    """
    input_tensor = Input(shape=input_shape)
    # x = CosDense(classes,activation=activation, name='cosine_dense')(feature_extracted)
    x = CosDense(classes, activation=activation, name='cosine_dense1')(input_tensor)
    model = Model(input_tensor, x, name='cos_classifier')
    return model
